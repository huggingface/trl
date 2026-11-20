"""List, and with --cancel cancel, my running HF Sandbox jobs that run the W2 image.

    python sweep_sandboxes.py                # list
    python sweep_sandboxes.py --cancel       # cancel them (only when no W2 driver is running: age doesn't tell an
                                             # orphan from a live sandbox, so this cancels every match)

Killing a driver (scancel, OOM, crash) does not tear down the sandboxes its rollouts created; they keep running and
billing until their 10 min idle timeout, or forever while a command is still running in them. Sweep twice: a
shutting-down server can start a few more seconds after the first pass. Matches on the sandbox label and the image
(`hf.co/spaces/<space>` shows up as `space_id`), so other users of HF Jobs in the namespace are never touched.
"""

import argparse
import datetime
import time

from huggingface_hub import HfApi


p = argparse.ArgumentParser()
p.add_argument("--space", default="aminediroHF/termgrade-base")
p.add_argument("--cancel", action="store_true")
p.add_argument("--passes", type=int, default=2)
p.add_argument(
    "--min-age-min",
    type=float,
    default=0,
    help="only sandboxes older than this (orphans of a stopped job while another W2 job runs)",
)
args = p.parse_args()
api = HfApi()


def matches():
    found = []
    # Filtered server-side: listing the whole job history pages for >10 min.
    for job in api.list_jobs(status=["RUNNING", "SCHEDULING"]):
        labels = job.labels or {}
        image = f"{job.space_id or ''} {job.docker_image or ''}"
        if "hf-sandbox" in labels and args.space in image:
            age_min = (
                (datetime.datetime.now(datetime.UTC) - job.created_at).total_seconds() / 60 if job.created_at else 1e9
            )
            if age_min >= args.min_age_min:
                found.append(job)
    return found


for i in range(args.passes if args.cancel else 1):
    jobs = matches()
    print(f"pass {i + 1}: {len(jobs)} running sandbox jobs on {args.space}", flush=True)
    if not args.cancel:
        for job in jobs:
            print(job.id, job.status.stage, job.created_at, job.flavor)
        break
    for job in jobs:
        try:
            api.cancel_job(job_id=job.id)
        except Exception as e:  # noqa: BLE001 - keep sweeping
            print("cancel failed", job.id, repr(e))
    if i + 1 < args.passes:
        time.sleep(30)
print("remaining:", len(matches()))
