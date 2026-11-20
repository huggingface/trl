"""Call reward_safe.compute_score from a worker thread (the dangerous case) and check it never blocks the caller."""

import sys
import threading
import time


sys.path.insert(0, sys.argv[1])
import reward
import reward_safe


def main():
    cases = [
        ("so \\boxed{9^{9^{9}}}", "3"),
        ("\\boxed{10^{10^{10}}}", "1"),
        ("\\boxed{99999999999^{99999999999}}", "2"),
        ("The answer is \\boxed{42}", "42"),
        ("\\boxed{\\frac{1}{2}}", "0.5"),
        ("\\boxed{7}", "8"),
    ]
    ticks = [0]
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.01)

    threading.Thread(target=ticker, daemon=True).start()
    out = {}

    def worker():
        for c, a in cases:
            t0 = time.monotonic()
            t_before = ticks[0]
            s = reward_safe.compute_score(c, a)
            dt = time.monotonic() - t0
            out[c] = (s, round(dt, 1), ticks[0] - t_before)

    th = threading.Thread(target=worker)
    th.start()
    th.join(300)
    stop.set()
    for c, a in cases:
        print(
            repr(c[-30:]),
            "->",
            out.get(c),
            "| main-thread reference:",
            reward.compute_score(c, a) if "^" not in c else "skipped",
        )
    print("ALL RETURNED" if len(out) == len(cases) else f"ONLY {len(out)}/{len(cases)} RETURNED")


if __name__ == "__main__":
    main()
