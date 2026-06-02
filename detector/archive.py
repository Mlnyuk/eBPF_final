#!/usr/bin/env python3
"""
archive.py
==========
Durable feature archive for the detector.

Every scored feature vector (input features + score + decision) is appended to a
daily-rotated CSV on a persistent volume, so the labelled stream survives pod
restarts and accumulates a retraining corpus (the basis for drift detection /
periodic retraining).

Design
------
* Disabled unless ARCHIVE_DIR is set (local/dev and tests are unaffected).
* One file per UTC-day PER REPLICA: ``features-YYYYMMDD-<host>.csv``. With the
  detector running 2+ replicas on a shared ReadWriteMany volume, per-host
  filenames avoid interleaved writes to the same file — no locking across pods.
* Thread-safe within a process (multiple uvicorn workers share one writer lock).
* ``extrasaction="ignore"`` so result rows carrying extra keys (e.g.
  top_features) don't break the fixed schema.
"""
from __future__ import annotations

import csv
import os
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Sequence


class FeatureArchive:
    def __init__(self, directory: str, fieldnames: Sequence[str]) -> None:
        self.enabled = bool(directory)
        self.fieldnames = list(fieldnames)
        self._lock = threading.Lock()
        self.host = os.environ.get("HOSTNAME") or socket.gethostname()
        self._dir = Path(directory) if directory else None
        if self.enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        day = time.strftime("%Y%m%d", time.gmtime())
        return self._dir / f"features-{day}-{self.host}.csv"

    def append(self, records: List[Dict]) -> None:
        """Append result records (input features + score columns) to today's file."""
        if not self.enabled or not records:
            return
        path = self._path()
        write_header = not path.exists()
        try:
            with self._lock:
                with path.open("a", newline="") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=self.fieldnames, extrasaction="ignore")
                    if write_header:
                        writer.writeheader()
                    writer.writerows(records)
        except OSError:
            # Archiving must never break scoring; drop silently on IO error.
            pass
