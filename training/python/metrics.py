"""Optional structured metrics output for the trainer.

Console logging is the default and always on. This adds two *opt-in*
machine-readable sinks, driven by `train.py --metrics-csv` /
`--tensorboard`:

  * a CSV file - one row per logged update, header written once, appended
    to on resume (dependency-free); and
  * a TensorBoard event dir - only if `tensorboard` is installed; a clear
    one-line hint is logged instead of crashing when it isn't.

`MetricsWriter` fans a single `log(step, {...})` call out to whichever
sinks are configured; with neither flag set it's a zero-cost no-op.
"""

from __future__ import annotations

import csv
import os
import sys

from logging_setup import get_logger

log = get_logger(__name__)


class MetricsWriter:
    def __init__(self, csv_path: str | None = None, tensorboard_dir: str | None = None) -> None:
        self._csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: list[str] | None = None
        self._tb = None

        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            # Resume-friendly: if the file already has a header, keep appending
            # under it and reuse its column order.
            existing_header = None
            if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
                with open(csv_path, newline="") as f:
                    existing_header = next(csv.reader(f), None)
            self._csv_file = open(csv_path, "a", newline="")
            if existing_header:
                self._csv_fields = existing_header
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=existing_header,
                                                  extrasaction="ignore")
            log.info("writing training metrics CSV to %s", csv_path)

        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                log.warning(
                    "--tensorboard given but the `tensorboard` package isn't installed; "
                    "skipping (install it with: %s -m pip install tensorboard)",
                    os.path.basename(sys.executable),
                )
            else:
                os.makedirs(tensorboard_dir, exist_ok=True)
                self._tb = SummaryWriter(tensorboard_dir)
                log.info("writing TensorBoard events to %s", tensorboard_dir)

    @property
    def active(self) -> bool:
        return self._csv_file is not None or self._tb is not None

    def log(self, step: int, values: dict[str, float]) -> None:
        """Record one row of `{metric: value}` at `step` (the PPO update
        number). NaNs are fine - they're written through as empty CSV cells
        and skipped for TensorBoard."""
        if self._csv_file is not None:
            row = {"update": step, **values}
            if self._csv_writer is None:  # first row of a brand-new file
                self._csv_fields = list(row.keys())
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields,
                                                  extrasaction="ignore")
                self._csv_writer.writeheader()
            self._csv_writer.writerow({k: _fmt(row.get(k)) for k in self._csv_fields})
            self._csv_file.flush()

        if self._tb is not None:
            for k, v in values.items():
                if v is not None and v == v:  # not None, not NaN
                    self._tb.add_scalar(k, float(v), step)
            self._tb.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None


def _fmt(v):
    """CSV cell: blank for missing/NaN, plain number otherwise."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return v
