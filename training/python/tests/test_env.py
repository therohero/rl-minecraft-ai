"""Tests for the pure wire-framing helpers in `env.py` (no running sim
needed): datagram fragmentation must round-trip and stay in lockstep with
`sim/src/protocol.rs`."""

import pytest

import env
from env import _HEADER, _build_frames, MAX_PAYLOAD, MSG_ACTION, WIRE_VERSION


def test_small_payload_is_one_frame():
    frames = _build_frames(MSG_ACTION, seq=42, payload=b"hello")
    assert len(frames) == 1
    mtype, ver, seq, idx, count = _HEADER.unpack(frames[0][: _HEADER.size])
    assert (mtype, ver, seq, idx, count) == (MSG_ACTION, WIRE_VERSION, 42, 0, 1)
    assert frames[0][_HEADER.size :] == b"hello"


def test_empty_payload_still_emits_one_frame():
    frames = _build_frames(MSG_ACTION, seq=1, payload=b"")
    assert len(frames) == 1
    assert frames[0][_HEADER.size :] == b""


def test_large_payload_fragments_and_reassembles():
    payload = bytes(i % 251 for i in range(MAX_PAYLOAD * 3 + 17))
    frames = _build_frames(MSG_ACTION, seq=7, payload=payload)
    assert len(frames) == 4

    rebuilt = b""
    for expected_idx, frame in enumerate(frames):
        mtype, ver, seq, idx, count = _HEADER.unpack(frame[: _HEADER.size])
        assert (mtype, ver, seq, count) == (MSG_ACTION, WIRE_VERSION, 7, 4)
        assert idx == expected_idx
        assert len(frame) - _HEADER.size <= MAX_PAYLOAD
        rebuilt += frame[_HEADER.size :]
    assert rebuilt == payload


def test_header_struct_is_ten_bytes():
    assert _HEADER.size == env.HEADER_LEN == 10


def test_zero_arenas_rejected():
    with pytest.raises(ValueError):
        env.SelfPlayArenaEnv(num_arenas=0, launch_sim=False)
