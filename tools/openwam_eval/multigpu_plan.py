"""Case ownership and measured environment steps for serial-model evaluation."""

import json
from pathlib import Path


def pending_keys(plan, confirmed, active):
    expected = {(s["seed"], i) for s in plan["seeds"] for i in s["layout_ids"]}
    assert set(confirmed) <= expected
    owned = set()
    for q in active:
        keys = {(q["seed"], i) for i in q["layout_ids"]}
        assert keys <= expected and not owned & keys, "Overlapping active cases"
        owned |= keys
    return expected - set(confirmed) - owned


class MetricSteps:
    """Sum actual per-env counter deltas, including resets; never infer from RPCs."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.counts = []
        self.total = 0
        self.parse_errors = 0
        self.truncations = 0

    def consume(self, event):
        if event.get("kind") == "reset":
            self.counts = []
            return 0
        if event.get("kind") != "take_action_batch":
            return 0
        counts = event.get("counts")
        if not isinstance(counts, list):
            return 0
        assert all(isinstance(n, int) and n >= 0 for n in counts)
        previous = self.counts if len(self.counts) == len(counts) else [0] * len(counts)
        delta = sum(n - p if n >= p else n for n, p in zip(counts, previous))
        self.counts = counts
        self.total += delta
        return delta

    def poll(self):
        if not self.path.exists():
            return 0
        before = self.total
        with self.path.open("rb") as f:
            if self.path.stat().st_size < self.offset:
                self.offset = 0
                self.counts = []
                self.truncations += 1
            f.seek(self.offset)
            while True:
                position = f.tell()
                line = f.readline()
                if not line or not line.endswith(b"\n"):
                    self.offset = position
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self.parse_errors += 1
                    self.offset = f.tell()
                    continue
                self.consume(event)
                self.offset = f.tell()
        return self.total - before
