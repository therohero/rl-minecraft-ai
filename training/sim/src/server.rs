//! Binary-over-UDP server that drives N arenas in lockstep with a single
//! Python client. There is deliberately no sleep/tick-rate anywhere in this
//! loop: it steps every arena and replies as fast as it can read the next
//! action batch and write a state batch back.
//!
//! See `protocol.rs` for the datagram framing. The exchange is strictly
//! request/response with a sequence number; a retransmitted request (same
//! sequence number) replays the cached previous reply instead of stepping
//! the arenas again.

use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};

use log::{error, info, warn};
use rand::Rng;
use rayon::prelude::*;

use crate::arena::Arena;
use crate::physics::DT;
use crate::protocol::{
    obs_floats_per_slot, Action, Hello, Observation, ACTION_FLOATS_PER_SLOT, HEADER_LEN, MAX_PAYLOAD,
    MSG_ACTION, MSG_HELLO_REQ, MSG_HELLO_RESP, MSG_STATE, WIRE_VERSION,
};

/// Below this arena count rayon's fork/join costs more than it saves.
const PARALLEL_THRESHOLD: usize = 8;

pub fn run(port: u16, num_arenas: usize, seed: Option<u64>) -> std::io::Result<()> {
    if num_arenas == 0 {
        error!("num_arenas must be >= 1, got 0");
        std::process::exit(1);
    }

    let base_seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
    info!("base_seed={base_seed} (pass --seed <n> to reproduce this exact run)");

    let socket = UdpSocket::bind(("127.0.0.1", port)).map_err(|e| {
        error!("failed to bind udp 127.0.0.1:{port}: {e}");
        e
    })?;

    // One state batch is several 60 KB datagrams sent back to back; give the
    // send buffer room so a burst never blocks on a slow reader. The kernel
    // clamps to net.core.wmem_max; a clamp is harmless here (loopback), it
    // just means very large arena counts lean on the retransmit path.
    {
        let sref = socket2::SockRef::from(&socket);
        let _ = sref.set_send_buffer_size(16 * 1024 * 1024);
        let _ = sref.set_recv_buffer_size(4 * 1024 * 1024);
    }

    let cfg = crate::config::cfg();
    let players_per_arena = cfg.players_per_arena();
    let obs_floats = obs_floats_per_slot(cfg);
    info!(
        "listening on udp 127.0.0.1:{port} with {num_arenas} arena(s), {}v{} ({} slots/arena), \
         kit={:?}, waiting for trainer...",
        cfg.team_size, cfg.team_size, players_per_arena, cfg.kit
    );

    let kit_name = match cfg.kit {
        crate::config::Kit::Sword => "sword",
        crate::config::Kit::Axe => "axe",
        crate::config::Kit::Uhc => "uhc",
    };
    let hello = Hello {
        num_arenas,
        players_per_arena,
        team_size: cfg.team_size,
        kit: kit_name.to_string(),
        attribute_swapping: cfg.attribute_swapping,
        natural_regen: cfg.natural_regen,
        friendly_fire: cfg.friendly_fire,
        max_observed_enemies: cfg.max_observed_enemies,
        max_observed_teammates: cfg.max_observed_teammates,
        max_observed_projectiles: cfg.max_observed_projectiles,
        block_view_size: cfg.block_view_size,
        item_count: crate::kit::ITEM_COUNT,
        hotbar_slots: crate::kit::HOTBAR_SLOTS,
        hotbar_action_dim: crate::kit::HOTBAR_ACTION_DIM,
        obs_floats_per_slot: obs_floats,
        action_floats_per_slot: ACTION_FLOATS_PER_SLOT,
        tick_dt: DT,
        max_hp: crate::combat::MAX_HP,
        arena_radius: cfg.arena_radius,
        match_time_seconds: cfg.match_time_seconds,
        terrain_max_amplitude: cfg.terrain_max_amplitude,
        max_look_delta: cfg.max_look_delta,
        max_ping_ms: cfg.max_ping_ms,
        config: cfg.clone(),
    };
    let hello_bytes = serde_json::to_vec(&hello).expect("Hello serializes");

    let mut session = Session::new(num_arenas, players_per_arena, obs_floats, base_seed);
    let mut recv_buf = vec![0u8; 65_536];

    loop {
        let (n, src) = match socket.recv_from(&mut recv_buf) {
            Ok(x) => x,
            Err(e) => {
                warn!("recv_from failed: {e}, continuing");
                continue;
            }
        };
        if n < HEADER_LEN {
            continue;
        }
        let msg_type = recv_buf[0];
        let version = recv_buf[1];
        if version != WIRE_VERSION {
            warn!("dropping datagram with wire version {version} (expected {WIRE_VERSION})");
            continue;
        }
        let seq = u32::from_le_bytes(recv_buf[2..6].try_into().unwrap());
        let frag_idx = u16::from_le_bytes(recv_buf[6..8].try_into().unwrap());
        let frag_count = u16::from_le_bytes(recv_buf[8..10].try_into().unwrap());
        let payload = &recv_buf[HEADER_LEN..n];

        match msg_type {
            MSG_HELLO_REQ => {
                info!("client hello from {src}, (re)starting session");
                session.reset();
                send_message(&socket, src, MSG_HELLO_RESP, 0, &hello_bytes);
            }
            MSG_ACTION => {
                if let Some(full_payload) =
                    session.ingest_fragment(seq, frag_idx, frag_count, payload)
                {
                    if let Some(frames) = session.response_for(seq, &full_payload) {
                        for f in frames {
                            let _ = socket.send_to(f, src);
                        }
                    }
                }
            }
            other => warn!("dropping datagram with unknown message type {other}"),
        }
    }
}

struct Session {
    num_arenas: usize,
    players_per_arena: usize,
    obs_floats: usize,
    base_seed: u64,
    arenas: Vec<Arena>,
    pending: HashMap<u32, PartialMessage>,
    last_seq: Option<u32>,
    last_frames: Vec<Vec<u8>>,
    /// Reused across steps: per-arena observation vectors, then the flat
    /// f32 scratch buffer, then the little-endian byte buffer.
    states: Vec<Vec<Observation>>,
    out_floats: Vec<f32>,
    out_bytes: Vec<u8>,
    steps: u64,
    start: std::time::Instant,
}

struct PartialMessage {
    frag_count: u16,
    have: u16,
    fragments: Vec<Vec<u8>>,
}

impl Session {
    fn new(num_arenas: usize, players_per_arena: usize, obs_floats: usize, base_seed: u64) -> Self {
        let num_slots = num_arenas * players_per_arena;
        let mut s = Session {
            num_arenas,
            players_per_arena,
            obs_floats,
            base_seed,
            arenas: Vec::new(),
            pending: HashMap::new(),
            last_seq: None,
            last_frames: Vec::new(),
            states: Vec::with_capacity(num_arenas),
            out_floats: Vec::with_capacity(num_slots * obs_floats),
            out_bytes: Vec::with_capacity(num_slots * obs_floats * 4),
            steps: 0,
            start: std::time::Instant::now(),
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.arenas = (0..self.num_arenas)
            .map(|i| {
                Arena::new(
                    self.base_seed
                        .wrapping_add(i as u64)
                        .wrapping_mul(0x9E3779B97F4A7C15),
                )
            })
            .collect();
        self.pending.clear();
        self.last_seq = None;
        self.last_frames.clear();
        self.steps = 0;
        self.start = std::time::Instant::now();
    }

    fn ingest_fragment(
        &mut self,
        seq: u32,
        frag_idx: u16,
        frag_count: u16,
        payload: &[u8],
    ) -> Option<Vec<u8>> {
        match self.last_seq {
            Some(l) if seq == l => return Some(Vec::new()),
            Some(l) if seq != l.wrapping_add(1) => {
                warn!("dropping stale/out-of-order action seq {seq} (last processed {l})");
                return None;
            }
            _ => {}
        }
        if frag_count == 0 || frag_idx >= frag_count {
            warn!("bad fragment header idx={frag_idx} count={frag_count}, dropping");
            return None;
        }

        let entry = self.pending.entry(seq).or_insert_with(|| PartialMessage {
            frag_count,
            have: 0,
            fragments: vec![Vec::new(); frag_count as usize],
        });
        if entry.frag_count != frag_count {
            warn!("fragment count changed mid-message for seq {seq}, resetting");
            *entry = PartialMessage {
                frag_count,
                have: 0,
                fragments: vec![Vec::new(); frag_count as usize],
            };
        }
        let slot = &mut entry.fragments[frag_idx as usize];
        if slot.is_empty() {
            *slot = payload.to_vec();
            entry.have += 1;
        }
        if entry.have != entry.frag_count {
            return None;
        }
        let msg = self.pending.remove(&seq).unwrap();
        Some(msg.fragments.concat())
    }

    fn response_for(&mut self, seq: u32, payload: &[u8]) -> Option<&[Vec<u8>]> {
        if Some(seq) == self.last_seq {
            return Some(&self.last_frames);
        }

        let num_slots = self.num_arenas * self.players_per_arena;
        let expected = num_slots * ACTION_FLOATS_PER_SLOT * 4;
        if payload.len() != expected {
            error!(
                "action payload is {} bytes, expected {} ({} slots x {} f32) - dropping",
                payload.len(),
                expected,
                num_slots,
                ACTION_FLOATS_PER_SLOT
            );
            return None;
        }

        let floats: Vec<f32> = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let ppa = self.players_per_arena;
        let slice_actions = |arena_i: usize| -> Vec<Action> {
            let base = arena_i * ppa * ACTION_FLOATS_PER_SLOT;
            (0..ppa)
                .map(|s| {
                    let o = base + s * ACTION_FLOATS_PER_SLOT;
                    Action::from_wire(&floats[o..o + ACTION_FLOATS_PER_SLOT])
                })
                .collect()
        };

        self.states.clear();
        if self.arenas.len() >= PARALLEL_THRESHOLD {
            self.states.par_extend(
                self.arenas
                    .par_iter_mut()
                    .enumerate()
                    .map(|(i, arena)| arena.step(&slice_actions(i))),
            );
        } else {
            for (i, arena) in self.arenas.iter_mut().enumerate() {
                let acts = slice_actions(i);
                self.states.push(arena.step(&acts));
            }
        }

        self.out_floats.clear();
        for arena_obs in &self.states {
            for obs in arena_obs {
                obs.write_wire(&mut self.out_floats);
            }
        }
        debug_assert_eq!(self.out_floats.len(), num_slots * self.obs_floats);
        self.out_bytes.clear();
        for v in &self.out_floats {
            self.out_bytes.extend_from_slice(&v.to_le_bytes());
        }

        self.last_frames = build_frames(MSG_STATE, seq, &self.out_bytes);
        self.last_seq = Some(seq);

        self.steps += 1;
        if self.steps.is_multiple_of(10_000) {
            let elapsed = self.start.elapsed().as_secs_f64();
            let sps = self.steps as f64 / elapsed;
            let env_sps = sps * num_slots as f64;
            info!(
                "{} batch-steps, {:.0} steps/sec (x{} slots = {:.0} env-steps/sec)",
                self.steps, sps, num_slots, env_sps
            );
        }

        Some(&self.last_frames)
    }
}

fn build_frames(msg_type: u8, seq: u32, payload: &[u8]) -> Vec<Vec<u8>> {
    let chunks: Vec<&[u8]> = if payload.is_empty() {
        vec![&[][..]]
    } else {
        payload.chunks(MAX_PAYLOAD).collect()
    };
    let frag_count = chunks.len() as u16;
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut frame = Vec::with_capacity(HEADER_LEN + chunk.len());
            frame.push(msg_type);
            frame.push(WIRE_VERSION);
            frame.extend_from_slice(&seq.to_le_bytes());
            frame.extend_from_slice(&(i as u16).to_le_bytes());
            frame.extend_from_slice(&frag_count.to_le_bytes());
            frame.extend_from_slice(chunk);
            frame
        })
        .collect()
}

fn send_message(socket: &UdpSocket, dst: SocketAddr, msg_type: u8, seq: u32, payload: &[u8]) {
    for frame in build_frames(msg_type, seq, payload) {
        if let Err(e) = socket.send_to(&frame, dst) {
            warn!("failed to send message type {msg_type} to {dst}: {e}");
            return;
        }
    }
}
