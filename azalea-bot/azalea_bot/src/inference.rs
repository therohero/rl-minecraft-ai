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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{info, warn};
use serde::Serialize;
use tokio::sync::watch;

use crate::{Action, LstmState, Observation};

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
///
/// For a recurrent (`--lstm`) policy, this loop is also where the carried
/// `(h, c)` state lives: it processes one observation at a time, in order,
/// so a plain local variable is enough - no extra synchronisation needed.
/// `episode_gen` is watched each iteration so the state gets zeroed at the
/// same point training does (see `State::episode_gen`'s doc comment); for
/// the default non-recurrent policy `lstm_state` just stays `None` and is
/// never sent.
pub(crate) async fn run_worker(
    http: reqwest::Client,
    act_url: Arc<String>,
    mut obs_rx: ObsReceiver,
    out: ActionCell,
    episode_gen: Arc<AtomicU64>,
) {
    let mut stats = RoundTripStats::new();
    let mut lstm_state: Option<LstmState> = None;
    let mut last_gen = episode_gen.load(Ordering::Relaxed);
    loop {
        if obs_rx.changed().await.is_err() {
            break; // every sender dropped - client is gone
        }
        let latest = obs_rx.borrow_and_update().clone();
        let Some((tick, observation)) = latest else {
            continue;
        };

        let gen = episode_gen.load(Ordering::Relaxed);
        if gen != last_gen {
            lstm_state = None;
            last_gen = gen;
        }

        let started = Instant::now();
        match fetch_action(&http, &act_url, &observation, lstm_state.clone()).await {
            Ok(action) => {
                stats.record(started.elapsed());
                // Carried into the next request whether or not this
                // checkpoint is recurrent - `None` stays `None`.
                lstm_state = action.lstm_state.clone();
                out.store(Decision { obs_tick: tick, action });
            }
            // Keep the last known state on a failed request (rather than
            // dropping to zeroed) and just retry with it next tick - same
            // "hold the last decision" spirit as the action itself below.
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

/// The `/act` request body: the observation's fields plus, only when
/// non-`None`, a top-level `lstm_state` - matches
/// `inference_server.py`'s contract (an absent/`null` `lstm_state` there
/// means "start from zeroed"). Generic over the flattened type so the wire
/// shape (flatten + skip-if-none) is unit-testable without a real,
/// ~40-field `Observation` (see `tests` below); `fetch_action` always
/// instantiates it as `ActRequest<'_, Observation>`.
#[derive(Serialize)]
struct ActRequest<'a, O: Serialize> {
    #[serde(flatten)]
    obs: &'a O,
    #[serde(skip_serializing_if = "Option::is_none")]
    lstm_state: Option<LstmState>,
}

async fn fetch_action(
    http: &reqwest::Client,
    act_url: &str,
    observation: &Observation,
    lstm_state: Option<LstmState>,
) -> eyre::Result<Action> {
    let body = ActRequest { obs: observation, lstm_state };
    let action = http
        .post(act_url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct FakeObs {
        a: i32,
        b: bool,
    }

    #[test]
    fn act_request_omits_lstm_state_when_none() {
        let obs = FakeObs { a: 1, b: true };
        let req = ActRequest { obs: &obs, lstm_state: None };
        let v = serde_json::to_value(&req).unwrap();
        // No `lstm_state` key at all - the non-recurrent wire shape must be
        // byte-for-byte what inference_server.py expected before lstm_state
        // existed.
        assert_eq!(v, serde_json::json!({"a": 1, "b": true}));
    }

    #[test]
    fn act_request_flattens_obs_and_includes_lstm_state_when_present() {
        // Exact in f32 (and so in the f64 JSON round-trip) - avoids a
        // binary-precision mismatch unrelated to what this test checks.
        let obs = FakeObs { a: 1, b: true };
        let state = LstmState { h: vec![0.5, 0.25], c: vec![-1.5, 2.0] };
        let req = ActRequest { obs: &obs, lstm_state: Some(state) };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], true);
        assert_eq!(v["lstm_state"]["h"], serde_json::json!([0.5, 0.25]));
        assert_eq!(v["lstm_state"]["c"], serde_json::json!([-1.5, 2.0]));
    }

    fn min_action_json() -> serde_json::Value {
        serde_json::json!({
            "move_x": 0.0, "move_z": 0.0, "yaw_delta": 0.0, "pitch_delta": 0.0,
            "jump": false, "attack": false, "sprint": false
        })
    }

    #[test]
    fn action_deserializes_without_lstm_state() {
        let a: Action = serde_json::from_value(min_action_json()).unwrap();
        assert!(a.lstm_state.is_none());
    }

    #[test]
    fn action_deserializes_with_lstm_state() {
        let mut json = min_action_json();
        json["lstm_state"] = serde_json::json!({"h": [1.0, 2.0], "c": [3.0]});
        let a: Action = serde_json::from_value(json).unwrap();
        let state = a.lstm_state.expect("lstm_state should have parsed");
        assert_eq!(state.h, vec![1.0, 2.0]);
        assert_eq!(state.c, vec![3.0]);
    }
}
