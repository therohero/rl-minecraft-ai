//! Off-the-tick-loop inference.
//!
//! The old bridge `await`ed the HTTP round-trip to the policy server
//! *inside* the game-tick handler, so a slow model or a hiccuping network
//! stalled the whole client - no movement, no look, no reaction - for as
//! long as the request took. Here the request runs on its own Tokio task:
//!
//!   * every tick the handler drops the freshest [`Observation`] into a
//!     `watch` channel (overwriting any the worker hasn't picked up yet)
//!     and immediately applies the most recent action it already has,
//!   * this worker loops: wait for a new observation, POST it, store the
//!     result in a shared cell.
//!
//! Net effect: the client acts every single 50 ms tick regardless of
//! inference latency, at the cost of the action being at most one
//! round-trip stale (which the tick handler tracks and warns about).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{info, warn};
use serde::Serialize;
use tokio::sync::watch;

use crate::{Action, Observation};

/// Everything one `/act` round-trip needs, snapshotted from `State` the
/// moment the tick's observation was built - live frame-stack history and
/// carried LSTM state, both no-ops for a plain memoryless policy (see
/// `Consts::frame_stack` / `Consts::lstm_hidden` in `main.rs`).
#[derive(Clone)]
pub(crate) struct ObsPacket {
    pub tick: u64,
    pub obs: Observation,
    /// Older frames, newest-first, length `0..=frame_stack-1` (fewer near
    /// the start of a life / just after a respawn - the server zero-pads
    /// the rest, mirroring `training/python/frame_stack.py::reset`).
    pub prev_frames: Vec<Observation>,
    /// Carried recurrent `(h, c)`, `Some` only for an `--lstm` policy.
    pub lstm: Option<(Vec<f32>, Vec<f32>)>,
}

/// An action together with the tick whose observation produced it, so the
/// tick loop can tell how stale the decision it is about to apply is.
#[derive(Clone)]
pub(crate) struct Decision {
    pub obs_tick: u64,
    pub action: Action,
}

/// Latest [`Decision`] from the inference worker, shared with the tick
/// handler. Cheap to clone (one `Arc`).
#[derive(Clone, Default)]
pub(crate) struct ActionCell(Arc<Mutex<Option<Decision>>>);

impl ActionCell {
    pub fn latest(&self) -> Option<Decision> {
        self.0.lock().unwrap().clone()
    }
    fn store(&self, decision: Decision) {
        *self.0.lock().unwrap() = Some(decision);
    }
}

/// The observation side of the channel - `None` before the first tick in a
/// loaded world.
pub(crate) type ObsSender = watch::Sender<Option<ObsPacket>>;
pub(crate) type ObsReceiver = watch::Receiver<Option<ObsPacket>>;

/// Runs until the observation channel closes (i.e. the client shut down).
pub(crate) async fn run_worker(
    http: reqwest::Client,
    act_url: Arc<String>,
    mut obs_rx: ObsReceiver,
    out: ActionCell,
) {
    let mut stats = RoundTripStats::new();
    loop {
        if obs_rx.changed().await.is_err() {
            break; // every sender dropped - client is gone
        }
        let latest = obs_rx.borrow_and_update().clone();
        let Some(packet) = latest else {
            continue;
        };
        let tick = packet.tick;
        let started = Instant::now();
        match fetch_action(&http, &act_url, &packet).await {
            Ok(action) => {
                stats.record(started.elapsed());
                out.store(Decision { obs_tick: tick, action });
            }
            Err(e) => warn!("inference request failed ({e}); holding last action"),
        }
    }
    info!("inference worker exiting (observation channel closed)");
}

/// Rolling summary of the `/act` round-trip so it's obvious whether the
/// HTTP+JSON bridge is actually a bottleneck (one 50 ms game tick is the
/// budget). Logged once every [`RoundTripStats::REPORT_EVERY`].
struct RoundTripStats {
    since_last_report: Instant,
    count: u32,
    sum: Duration,
    max: Duration,
    over_budget: u32,
}

impl RoundTripStats {
    const REPORT_EVERY: Duration = Duration::from_secs(30);
    /// A round-trip past this fraction of a tick is worth flagging.
    const BUDGET: Duration = Duration::from_millis(25);

    fn new() -> Self {
        RoundTripStats {
            since_last_report: Instant::now(),
            count: 0,
            sum: Duration::ZERO,
            max: Duration::ZERO,
            over_budget: 0,
        }
    }

    fn record(&mut self, elapsed: Duration) {
        self.count += 1;
        self.sum += elapsed;
        self.max = self.max.max(elapsed);
        if elapsed > Self::BUDGET {
            self.over_budget += 1;
        }
        if self.since_last_report.elapsed() >= Self::REPORT_EVERY && self.count > 0 {
            let mean_ms = self.sum.as_secs_f64() * 1e3 / self.count as f64;
            info!(
                "inference /act round-trip (last {}s): mean {:.1} ms, max {:.1} ms, {}/{} over {} ms",
                Self::REPORT_EVERY.as_secs(),
                mean_ms,
                self.max.as_secs_f64() * 1e3,
                self.over_budget,
                self.count,
                Self::BUDGET.as_millis(),
            );
            *self = RoundTripStats::new();
        }
    }
}

/// The `/act` request body: this tick's observation flattened at the top
/// level (unchanged from before frame-stacking/LSTM existed - a plain
/// memoryless policy's client is unaffected byte-for-byte), plus additive
/// optional fields a server that knows about them can use. `observation_to_row`
/// (`training/python/features.py`) only ever reads known keys, so an older
/// server ignoring these is exactly as compatible as one that understands
/// them - and a client never needs to send them for a `frame_stack`-1,
/// non-recurrent policy.
#[derive(Serialize)]
struct ActRequest<'a> {
    #[serde(flatten)]
    obs: &'a Observation,
    prev_frames: &'a [Observation],
    #[serde(skip_serializing_if = "Option::is_none")]
    lstm_h: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lstm_c: Option<&'a [f32]>,
}

async fn fetch_action(
    http: &reqwest::Client,
    act_url: &str,
    packet: &ObsPacket,
) -> eyre::Result<Action> {
    let req = ActRequest {
        obs: &packet.obs,
        prev_frames: &packet.prev_frames,
        lstm_h: packet.lstm.as_ref().map(|(h, _)| h.as_slice()),
        lstm_c: packet.lstm.as_ref().map(|(_, c)| c.as_slice()),
    };
    let action = http
        .post(act_url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(action)
}
