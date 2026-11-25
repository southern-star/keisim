"""Timing for a finished KeiPilot drive, with optional asynchronous KeiReality output."""
import numpy as np


def quantiles(values):
    if not len(values):
        return None
    p50, p95 = np.percentile(values, (50, 95))
    return {"p50": float(p50), "p95": float(p95)}


def summarize(drive, warmup=0):
    rows = drive.steps[warmup:]
    if len(rows) < 2:
        raise ValueError("Drive too short to measure")
    starts = np.array([r["start"] for r in rows])
    elapsed = starts[-1] - starts[0]
    work = np.array([r["work_ms"] for r in rows])
    # Count completed RGB images in the same wall-clock interval as control starts.
    completed = [r for r in drive.rendered if starts[0] <= r[1] < starts[-1]]
    dropped = len(drive.steps) - len(drive.rendered) if drive.mode != "pilot" else 0
    if dropped != drive.dropped or len({r[0] for r in drive.rendered}) != len(drive.rendered):
        raise ValueError("Incomplete display accounting")
    return dict(result=drive.result, ticks=len(drive.steps), measured_ticks=len(rows),
                wall_control_hz=(len(rows)-1)/elapsed, control_interval_ms=quantiles(np.diff(starts)*1000),
                control_work_ms=quantiles(work), over_100ms=int((work > 100).sum()),
                camera_ms=quantiles([r["camera_ms"] for r in rows]), pilot_ms=quantiles([r["pilot_ms"] for r in rows]),
                generated_frames=len(drive.rendered), skipped_frames=dropped,
                generated_fps=len(completed)/elapsed if drive.rendered else None,
                generation_work_ms=quantiles([r[2] for r in completed]),
                observation_to_rgb_ms=quantiles([r[3] for r in completed]))
