"""Unit tests for `evaluate.py`'s rating math and match scheduling - the
parts that don't need the Rust sim. The end-to-end run (real sim +
checkpoints -> Elo table) is exercised by `smoke_train.py`.
"""

import numpy as np

from evaluate import _build_queue, bradley_terry_elo, discover_ladder, resolve_checkpoint


def test_bradley_terry_orders_by_strength():
    # 3 players, A beats B every time, B beats C every time, A beats C every time.
    n = 3
    wins = np.zeros((n, n))
    wins[0, 1] = wins[1, 2] = wins[0, 2] = 20
    games = wins + wins.T
    elo = bradley_terry_elo(wins, games)
    assert elo[0] > elo[1] > elo[2]
    # mean is anchored to ~0
    assert abs(elo.mean()) < 1e-6


def test_bradley_terry_equal_results_equal_rating():
    n = 3
    credit = np.full((n, n), 10.0)
    np.fill_diagonal(credit, 0.0)
    games = np.full((n, n), 20.0)
    np.fill_diagonal(games, 0.0)
    elo = bradley_terry_elo(credit, games)
    assert np.allclose(elo, elo[0], atol=1e-6)


def test_bradley_terry_unbeaten_player_stays_finite():
    n = 2
    wins = np.array([[0.0, 30.0], [0.0, 0.0]])
    games = wins + wins.T
    elo = bradley_terry_elo(wins, games)
    assert np.isfinite(elo).all()
    assert elo[0] > elo[1]


def test_build_queue_counts_and_side_balance():
    q = _build_queue(n_players=4, matches_per_pair=10, rng=np.random.default_rng(0))
    assert len(q) == 6 * 10  # 4 choose 2 pairs
    # every unordered pair appears exactly matches_per_pair times, split 5/5.
    from collections import Counter
    ordered = Counter(q)
    for i in range(4):
        for j in range(i + 1, 4):
            assert ordered[(i, j)] + ordered[(j, i)] == 10
            assert ordered[(i, j)] == ordered[(j, i)] == 5


def test_discover_ladder_excludes_candidate_and_subsamples(tmp_path):
    for u in (100, 200, 300, 400, 500):
        (tmp_path / f"policy_update_{u}.pt").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    cand = str(tmp_path / "policy_update_300.pt")
    ladder = discover_ladder(str(tmp_path), cand, max_ladder=3)
    names = sorted(p.split("_")[-1] for p in ladder)
    assert "300.pt" not in names
    assert len(ladder) <= 3


def test_resolve_checkpoint_accepts_path_or_bare_name(tmp_path):
    p = tmp_path / "latest.pt"
    p.write_bytes(b"x")
    assert resolve_checkpoint(str(p), "/nonexistent") == str(p)
    assert resolve_checkpoint("latest.pt", str(tmp_path)) == str(p)
