"""Tests for `metrics.py` - the opt-in CSV / TensorBoard sink. The
end-to-end wiring (train.py actually emitting rows) is covered by
`smoke_train.py`."""

import csv
import math

from metrics import MetricsWriter


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_noop_when_nothing_configured():
    w = MetricsWriter()
    assert not w.active
    w.log(1, {"x": 1.0})  # must not raise
    w.close()


def test_csv_header_and_rows(tmp_path):
    p = str(tmp_path / "m.csv")
    w = MetricsWriter(csv_path=p)
    assert w.active
    w.log(10, {"loss": 1.5, "win_rate": 0.4})
    w.log(20, {"loss": 1.2, "win_rate": 0.5})
    w.close()
    rows = _rows(p)
    assert [r["update"] for r in rows] == ["10", "20"]
    assert rows[0]["loss"] == "1.5"
    assert rows[1]["win_rate"] == "0.5"


def test_csv_nan_becomes_blank(tmp_path):
    p = str(tmp_path / "m.csv")
    w = MetricsWriter(csv_path=p)
    w.log(1, {"win_rate": math.nan, "loss": 0.9})
    w.close()
    assert _rows(p)[0]["win_rate"] == ""


def test_csv_appends_under_existing_header_on_resume(tmp_path):
    p = str(tmp_path / "m.csv")
    w1 = MetricsWriter(csv_path=p)
    w1.log(1, {"a": 1, "b": 2})
    w1.close()

    # resume: same file, and a row that also carries an unknown column.
    w2 = MetricsWriter(csv_path=p)
    w2.log(2, {"a": 3, "b": 4, "c": 5})
    w2.close()

    rows = _rows(p)
    assert [r["update"] for r in rows] == ["1", "2"]
    assert "c" not in rows[0]  # header from the first session is kept
    assert rows[1]["a"] == "3"


def test_tensorboard_missing_package_is_not_fatal(tmp_path):
    # Whether or not `tensorboard` is installed, constructing must not raise.
    w = MetricsWriter(tensorboard_dir=str(tmp_path / "tb"))
    w.log(1, {"x": 1.0})
    w.close()
