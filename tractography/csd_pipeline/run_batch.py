"""Batch driver for the 18-subject A3 free-water tractography cohort.

Sequential by default; --parallel N runs N subjects concurrently via a
process pool (each subject still uses --nthreads threads internally, so
parallel * nthreads should stay well under the machine's core count).
Idempotent per-subject (run_subject.already_done), so this is safe to
interrupt and re-invoke -- completed subjects are skipped unless --force.
Continues past individual subject failures rather than aborting the batch.
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from paths import OUT_ROOT, SUBJECTS
from run_subject import already_done, run_subject


def _run_one(sid, force, nthreads, n_seeds):
    try:
        run_subject(sid, force=force, nthreads=nthreads, n_seeds=n_seeds)
        return sid, True, None
    except Exception as e:  # noqa: BLE001
        return sid, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=2_000_000)
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()

    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log = log_dir / f"batch_run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(batch_log), logging.StreamHandler(sys.stdout)],
    )

    todo = [s for s in args.subjects if args.force or not already_done(s)]
    skipped = [s for s in args.subjects if s not in todo]
    logging.info(f"batch: {len(todo)} to run, {len(skipped)} already done and skipped: {skipped}")

    t0 = time.time()
    results = []

    if args.parallel <= 1:
        for sid in todo:
            logging.info(f"[{sid}] starting")
            sid_, ok, err = _run_one(sid, args.force, args.nthreads, args.n_seeds)
            results.append((sid_, ok, err))
            if ok:
                logging.info(f"[{sid}] done")
            else:
                logging.error(f"[{sid}] FAILED: {err}")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_run_one, sid, args.force, args.nthreads, args.n_seeds): sid for sid in todo}
            for fut in as_completed(futs):
                sid_, ok, err = fut.result()
                results.append((sid_, ok, err))
                if ok:
                    logging.info(f"[{sid_}] done")
                else:
                    logging.error(f"[{sid_}] FAILED: {err}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    logging.info(f"batch complete in {time.time() - t0:.1f}s: {n_ok} ok, {n_fail} failed, {len(skipped)} skipped")
    if n_fail:
        logging.error(f"failed subjects: {[s for s, ok, _ in results if not ok]}")


if __name__ == "__main__":
    main()
