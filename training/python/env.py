"""Python-side client for the Rust simulation backend.

Launches `mc_pvp_sim` as a subprocess (or connects to one already running),
speaks the binary-over-UDP protocol defined in sim/src/protocol.rs, and
exposes a vectorized-env-style interface: `reset()` / `step(actions)` over
`num_arenas * players_per_arena` independent player-slots, where
`players_per_arena = 2 * team_size` (so 2 for a 1v1, 4 for a 2v2, ...) and
each arena's slots come out in team order. Because every observation is
built relative to "self" (and the nearest teammates/enemies), a single
shared policy controlling every slot IS self-play - no separate opponent
process needed.

Transport
---------
The hot loop exchanges one datagram (or a few fragments of one) per env
step: a flat little-endian ``float32`` action array out, a flat
``float32`` observation array back, decoded straight into a numpy buffer
with ``np.frombuffer`` - no per-element JSON parsing. The exchange is
strictly request/response tagged with a 32-bit sequence number; if a reply
doesn't arrive within ``RECV_TIMEOUT`` the same request is retransmitted
(the sim replays its cached reply for a duplicate sequence number rather
than stepping twice), so a rare loopback drop can't desync a run. Only the
one-time ``Hello`` handshake is still JSON.
"""

import json
import os
import socket
import struct
import subprocess
import time

import numpy as np

import features
from features import ACTION_FLOATS_PER_SLOT, wire_batch_to_obs
from logging_setup import get_logger

log = get_logger(__name__)

# --- wire framing: keep in lockstep with sim/src/protocol.rs ---
WIRE_VERSION = 9
MSG_HELLO_REQ = 1
MSG_HELLO_RESP = 2
MSG_ACTION = 3
MSG_STATE = 4
HEADER_LEN = 10
MAX_PAYLOAD = 60_000
_HEADER = struct.Struct("<BBIHH")  # type, version, seq, frag_idx, frag_count

# Seconds to wait for a reply datagram before retransmitting the request.
# On loopback a round-trip is microseconds, so this only fires on an actual
# datagram drop (a large state batch overflowing the socket buffer at high
# arena counts). Keep it well above any plausible GC / scheduler hiccup but
# far below the old 2.0 s - a drop then costs ~0.4 s to recover, not 2 s.
RECV_TIMEOUT = 0.4
MAX_RETRIES = 60

# `cargo build` names the binary `mc_pvp_sim.exe` on Windows and plain
# `mc_pvp_sim` on Linux/macOS.
DEFAULT_SIM_BINARY = "../sim/target/release/" + ("mc_pvp_sim.exe" if os.name == "nt" else "mc_pvp_sim")


def _build_frames(msg_type: int, seq: int, payload: bytes) -> list[bytes]:
    """Splits `payload` into datagrams of at most `MAX_PAYLOAD` bytes, each
    prefixed with the 10-byte header and sharing `seq`."""
    chunks = [payload[i : i + MAX_PAYLOAD] for i in range(0, len(payload), MAX_PAYLOAD)] or [b""]
    frag_count = len(chunks)
    return [
        _HEADER.pack(msg_type, WIRE_VERSION, seq, idx, frag_count) + chunk
        for idx, chunk in enumerate(chunks)
    ]


class SelfPlayArenaEnv:
    def __init__(
        self,
        num_arenas: int = 64,
        port: int = 9999,
        sim_binary: str = DEFAULT_SIM_BINARY,
        launch_sim: bool = True,
        connect_timeout: float = 10.0,
        sim_config_path: str | None = None,
        seed: int | None = None,
    ):
        if num_arenas < 1:
            raise ValueError(f"num_arenas must be >= 1, got {num_arenas}")

        self.num_arenas = num_arenas
        # Filled in from the Hello handshake below: 2 * team_size.
        self.players_per_arena = 2
        self.num_slots = num_arenas * 2
        self._proc = None
        self._seq = 0
        self.sim_config: dict | None = None

        if launch_sim:
            if not os.path.isfile(sim_binary):
                raise FileNotFoundError(
                    f"simulation binary not found at '{sim_binary}'. "
                    "Build it first: `cd sim && cargo build --release`, "
                    "or pass --sim-binary with the correct path."
                )
            cmd = [sim_binary, "--port", str(port), "--arenas", str(num_arenas)]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if sim_config_path is not None:
                if not os.path.isfile(sim_config_path):
                    raise FileNotFoundError(f"sim config file not found: '{sim_config_path}'")
                cmd += ["--config", sim_config_path]
            log.info("launching simulation backend: %s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd)

            # A port conflict (a stale mc_pvp_sim still bound to --port) makes
            # the freshly launched sim fail its UDP bind and exit within a few
            # ms. Catch that here with a clear message instead of silently
            # handshaking with whatever *else* is on the port.
            try:
                code = self._proc.wait(timeout=0.5)
                raise ConnectionError(
                    f"the simulation exited immediately (code {code}) - most likely "
                    f"another mc_pvp_sim is already bound to UDP port {port}. Stop it "
                    f"(e.g. `pkill -f mc_pvp_sim`) or pass a different --port."
                )
            except subprocess.TimeoutExpired:
                pass  # still running after the grace period - as expected

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # A single state batch is several 60 KB datagrams that arrive back to
        # back on loopback; the default ~208 KB socket buffer overflows and
        # drops fragments once the arena count is high, forcing a 2 s
        # retransmit stall. Ask for plenty - the kernel silently clamps to
        # net.core.rmem_max / wmem_max, and we check the effective size below.
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, opt, 16 * 1024 * 1024)
            except OSError:
                pass
        self._sock.connect(("127.0.0.1", port))
        self._sock.settimeout(RECV_TIMEOUT)

        hello = self._handshake(connect_timeout)

        if hello["num_arenas"] != num_arenas:
            raise RuntimeError(f"sim reports {hello['num_arenas']} arenas, expected {num_arenas}")
        self.tick_dt = hello["tick_dt"]
        self.sim_config = hello.get("config")
        self.players_per_arena = hello.get("players_per_arena", 2)
        self.team_size = hello.get("team_size", 1)
        self.kit = hello.get("kit", "sword")
        self.num_slots = num_arenas * self.players_per_arena

        item_count = hello.get("item_count", features.ITEM_COUNT)
        if item_count != features.ITEM_COUNT:
            raise RuntimeError(
                f"sim reports item_count={item_count} but features.py has {features.ITEM_COUNT} "
                "- kit.rs and features.py are out of sync"
            )
        hotbar_slots = hello.get("hotbar_slots", features.HOTBAR_SLOTS)
        if hotbar_slots != features.HOTBAR_SLOTS:
            raise RuntimeError(
                f"sim reports hotbar_slots={hotbar_slots} but features.py has "
                f"{features.HOTBAR_SLOTS} - kit.rs and features.py are out of sync"
            )
        hotbar_action_dim = hello.get("hotbar_action_dim", features.HOTBAR_ACTION_DIM)
        if hotbar_action_dim != features.HOTBAR_ACTION_DIM:
            raise RuntimeError(
                f"sim reports hotbar_action_dim={hotbar_action_dim} but features.py has "
                f"{features.HOTBAR_ACTION_DIM} - kit.rs and features.py are out of sync"
            )
        effect_count = hello.get("effect_count", features.EFFECT_COUNT)
        if effect_count != features.EFFECT_COUNT:
            raise RuntimeError(
                f"sim reports effect_count={effect_count} but features.py has "
                f"{features.EFFECT_COUNT} - effects.rs and features.py are out of sync"
            )

        features.configure(
            max_hp=hello.get("max_hp", 20.0),
            match_time_seconds=hello.get("match_time_seconds", 90.0),
            arena_radius=hello.get("arena_radius", 12.0),
            terrain_max_amplitude=hello.get("terrain_max_amplitude", 3.0),
            max_look_delta=hello.get("max_look_delta", 3.0),
            max_ping_ms=hello.get("max_ping_ms", 100.0),
            max_observed_enemies=hello.get("max_observed_enemies", 3),
            max_observed_teammates=hello.get("max_observed_teammates", 2),
            max_observed_projectiles=hello.get("max_observed_projectiles", 2),
            block_view_size=hello.get("block_view_size", 5),
        )
        wire_w = hello.get("obs_floats_per_slot")
        if wire_w is not None and wire_w != features.WIRE_FLOATS_PER_SLOT:
            raise RuntimeError(
                f"sim says obs is {wire_w} floats/slot but features.py computes "
                f"{features.WIRE_FLOATS_PER_SLOT} - protocol.rs and features.py are out of sync"
            )

        # Warn if the OS clamped our receive buffer close to (or below) one
        # full state batch - then a high arena count will hit retransmit
        # stalls. Linux getsockopt reports 2x the usable size, so compare the
        # raw value against 2x the batch and report half of it.
        state_bytes = self.num_slots * features.WIRE_FLOATS_PER_SLOT * 4
        rcvbuf = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if rcvbuf < 3 * state_bytes:
            log.warning(
                "UDP receive buffer is ~%d KB, barely above one %d KB state batch - the kernel "
                "clamped SO_RCVBUF to net.core.rmem_max. If you see 'retransmitting step' "
                "warnings (or raise --num-arenas), lift it once with: "
                "sudo sysctl -w net.core.rmem_max=16777216",
                rcvbuf // 2048,
                state_bytes // 1024,
            )
        log.info(
            "connected to simulation over UDP: %d arenas x %dv%d (%d slots), kit=%s, "
            "tick_dt=%.4fs, obs_dim=%d, constants=%s",
            num_arenas,
            self.team_size,
            self.team_size,
            self.num_slots,
            self.kit,
            self.tick_dt,
            features.OBS_DIM,
            features.active_constants(),
        )

    def _handshake(self, connect_timeout: float) -> dict:
        deadline = time.time() + connect_timeout
        req = _build_frames(MSG_HELLO_REQ, 0, b"")[0]
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                self._sock.send(req)
                payload = self._recv_message(MSG_HELLO_RESP, expected_seq=None)
            except (socket.timeout, ConnectionError) as e:
                last_err = e
                self._check_process_alive()
                time.sleep(0.1)
                continue
            # Got a reply - but if the sim *we* launched has already exited,
            # that reply came from some other process on this port (a stale
            # mc_pvp_sim that failed to be cleaned up). Never silently attach
            # to an unknown sim: its config / seed / build may not match.
            if self._proc is not None and self._proc.poll() is not None:
                port = self._sock.getpeername()[1]
                raise ConnectionError(
                    f"the simulation this trainer launched exited immediately "
                    f"(code {self._proc.returncode}) yet a handshake reply still came "
                    f"back on UDP port {port} - another mc_pvp_sim is already bound "
                    f"there. Stop it (e.g. `pkill -f mc_pvp_sim`) or pass a different "
                    f"--port. Not attaching to an unknown sim."
                )
            return json.loads(payload)
        self._fail_startup("never received the simulation's handshake")
        raise ConnectionError(
            f"timed out waiting for the simulation's UDP handshake on port {self._sock.getpeername()[1]} "
            f"({last_err}) - if another sim instance is already bound to this port, stop it or pass a "
            "different --port"
        )

    def _recv_message(self, expected_type: int, expected_seq: int | None) -> bytes:
        """Receives datagrams until every fragment of one `expected_type`
        message (matching `expected_seq`, if given) has arrived, then
        returns the reassembled payload. Datagrams that don't match are
        ignored (stale retransmits, wrong type)."""
        fragments: dict[int, bytes] = {}
        frag_count: int | None = None
        while True:
            data = self._sock.recv(65_536)
            if len(data) < HEADER_LEN:
                continue
            msg_type, version, seq, frag_idx, count = _HEADER.unpack_from(data)
            if version != WIRE_VERSION or msg_type != expected_type:
                continue
            if expected_seq is not None and seq != expected_seq:
                continue
            frag_count = count
            fragments[frag_idx] = data[HEADER_LEN:]
            if len(fragments) == frag_count:
                return b"".join(fragments[i] for i in range(frag_count))

    def _txn(self, action_bytes: bytes) -> bytes:
        """Sends one action batch and returns the reassembled state payload,
        retransmitting on timeout."""
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        frames = _build_frames(MSG_ACTION, self._seq, action_bytes)
        for attempt in range(MAX_RETRIES):
            for f in frames:
                self._sock.send(f)
            try:
                return self._recv_message(MSG_STATE, expected_seq=self._seq)
            except socket.timeout:
                self._check_process_alive()
                if attempt == MAX_RETRIES - 1:
                    raise ConnectionError(
                        f"no reply from the simulation after {MAX_RETRIES} retransmits of step {self._seq}"
                    )
                log.warning("no reply for step %d (attempt %d), retransmitting", self._seq, attempt + 1)
        raise AssertionError("unreachable")

    def reset(self):
        """Sends a no-op action batch to prime the first observation."""
        noop = np.zeros((self.num_slots, ACTION_FLOATS_PER_SLOT), dtype=np.float32)
        return self._step_from_array(noop)[0]

    def step(self, actions):
        """`actions` is a flat sequence of length `num_slots` (arena0.a,
        arena0.b, arena1.a, ...), each an `ACTION_FLOATS_PER_SLOT`-element
        sequence in the order [move_x, move_z, yaw_delta, pitch_delta, jump,
        attack, sprint, use_item, sneak, held_slot], OR an ndarray of shape
        [num_slots, ACTION_FLOATS_PER_SLOT] (preferred - avoids a copy).

        Returns (obs, rewards, dones, info):
          obs:     float32 array [num_slots, OBS_DIM]
          rewards: float32 array [num_slots]
          dones:   bool    array [num_slots]
          info:    dict of vectorized per-slot arrays: {"won", "lost", "raw"}
                   ("raw" is the [num_slots, WIRE_FLOATS_PER_SLOT] decoded
                   wire array, for custom logging / reward shaping)
        """
        arr = np.ascontiguousarray(actions, dtype=np.float32)
        if arr.shape != (self.num_slots, ACTION_FLOATS_PER_SLOT):
            raise ValueError(
                f"expected actions of shape ({self.num_slots}, {ACTION_FLOATS_PER_SLOT}), got {arr.shape}"
            )
        return self._step_from_array(arr)

    def _step_from_array(self, arr: np.ndarray):
        payload = self._txn(arr.tobytes())
        raw = np.frombuffer(payload, dtype="<f4")
        wire_w = features.WIRE_FLOATS_PER_SLOT
        expected = self.num_slots * wire_w
        if raw.size != expected:
            raise ConnectionError(
                f"state payload had {raw.size} floats, expected {expected} "
                f"({self.num_slots} slots x {wire_w})"
            )
        raw = raw.reshape(self.num_slots, wire_w)

        obs = wire_batch_to_obs(raw)
        rewards = raw[:, features.WIRE_REWARD].astype(np.float32)
        dones = raw[:, features.WIRE_DONE] != 0.0
        info = {
            "won": raw[:, features.WIRE_WON] != 0.0,
            "lost": raw[:, features.WIRE_LOST] != 0.0,
            "raw": raw,
        }

        if not np.isfinite(obs).all():
            log.error("non-finite values in observation batch - check reward/physics for NaN/Inf sources")

        return obs, rewards, dones, info

    def _check_process_alive(self) -> None:
        if self._proc is not None:
            code = self._proc.poll()
            if code is not None:
                log.error("simulation process has exited with code %s", code)

    def _fail_startup(self, message: str) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            log.error(
                "%s (simulation process already exited with code %s - see its log output above)",
                message,
                self._proc.returncode,
            )
        else:
            log.error(message)

    def close(self):
        log.info("closing environment")
        try:
            self._sock.close()
        except Exception as e:  # noqa: BLE001
            log.warning("error closing socket: %s", e)
        finally:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log.warning("simulation process did not exit in time, killing it")
                    self._proc.kill()
