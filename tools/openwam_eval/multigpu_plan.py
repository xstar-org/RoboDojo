"""Case ownership and measured environment steps for serial-model evaluation."""
import json
from collections import Counter
from pathlib import Path


def slot_specs(cpus, gpus, workers_per_gpu=2, cores_per_worker=6):
    assert len(set(cpus)) == len(cpus)
    assert len(set(gpus)) == len(gpus) == 3
    count = len(gpus) * workers_per_gpu
    assert count * cores_per_worker <= len(cpus) - 8
    return [dict(slot=i, gpu=gpus[i // workers_per_gpu],
                 cpus=cpus[i * cores_per_worker:(i + 1) * cores_per_worker])
            for i in range(count)]


def pending_keys(plan, confirmed, active):
    expected = {(s['seed'], i) for s in plan['seeds'] for i in s['layout_ids']}
    assert set(confirmed) <= expected
    owned = set()
    for q in active:
        keys = {(q['seed'], i) for i in q['layout_ids']}
        assert keys <= expected and not owned & keys, 'Overlapping active cases'
        owned |= keys
    return expected - set(confirmed) - owned


def next_assignment(plans, confirmed, active, priority, batch=5):
    counts = Counter(q['task'] for q in active)
    candidates = []
    for rank, task in enumerate(priority):
        requests = [q for q in active if q['task'] == task]
        pending = pending_keys(plans[task], confirmed[task], requests)
        if pending:
            candidates.append((counts[task], rank, task, pending))
    if not candidates:
        return None
    _, _, task, pending = min(candidates)
    limit = batch[task] if isinstance(batch, dict) else batch
    assert limit > 0
    for seedplan in plans[task]['seeds']:
        seed = seedplan['seed']
        ids = [i for i in seedplan['layout_ids'] if (seed, i) in pending][:limit]
        if ids:
            return task, seed, ids
    raise AssertionError('Pending case cannot be assigned')


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
        if event.get('kind') == 'reset':
            self.counts = []
            return 0
        if event.get('kind') != 'take_action_batch':
            return 0
        counts = event.get('counts')
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
        with self.path.open('rb') as f:
            if self.path.stat().st_size < self.offset:
                self.offset = 0
                self.counts = []
                self.truncations += 1
            f.seek(self.offset)
            while True:
                position = f.tell()
                line = f.readline()
                if not line or not line.endswith(b'\n'):
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
