"""Respect RoboDojo's native exclusion of unstable episodes from score denominators."""

import re

COUNTS = re.compile(r"Success nums:\s*(\d+),\s*Fail nums:\s*(\d+),\s*Unstable nums:\s*(\d+)")


def collect_attempt(rows, excluded, *, worker, seed, requested, data, log, attempt):
    requested = set(requested)
    details = list((data or {}).get("details", {}).values())
    for item in details:
        layout = item["layout_id"]
        assert layout in requested
        key = (seed, layout)
        if key not in excluded:
            rows.setdefault(key, dict(seed=seed, worker=worker, **item))
    counts = COUNTS.findall(log)
    if not counts:
        return False
    successes, failures, unstable = map(int, counts[-1])
    scored = {item["layout_id"] for item in details}
    # Only a complete native round can identify exclusions. A missing reply,
    # exception, timeout, or incomplete result is still an infrastructure retry.
    if successes + failures + unstable != len(requested) or successes + failures != len(details):
        return False
    if data is not None and data.get("eval_time") != len(details):
        return False
    missing = requested - scored
    if len(missing) != unstable:
        return False
    for layout in missing:
        key = (seed, layout)
        if key not in rows:
            excluded[key] = dict(seed=seed, layout_id=layout, worker=worker, reason="native_unstable", attempt=attempt)
    return True
