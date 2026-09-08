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
use tokio::sync::watch;

use crate::{Action, Observation};

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

/// The observation side of the channel: `(tick, observation)`, `None`
/// before the first tick in a loaded world.
pub(crate) type ObsSender = watch::Sender<Option<(u64, Observation)>>;
pub(crate) type ObsReceiver = watch::Receiver<Option<(u64, Observation)>>;

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
        let Some((tick, observation)) = latest else {
            continue;
        };
        let started = Instant::now();
        match fetch_action(&http, &act_url, &observation).await {
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

async fn fetch_action(
    http: &reqwest::Client,
    act_url: &str,
    observation: &Observation,
) -> eyre::Result<Action> {
    let action = http
        .post(act_url)
        .json(observation)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(action)
}
