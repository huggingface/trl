"""W1 math reward: 1.0 if math_verify judges the last \\boxed{} of the completion equal to the answer, else 0.0.

Depends only on `math-verify` (install `math-verify[antlr4_13_2]`). Every library calls `compute_score` or matches it.

    from reward import compute_score
    compute_score("... so the answer is \\boxed{42}.", "42")  # 1.0
"""

import threading

from math_verify import LatexExtractionConfig, parse, verify


PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def last_boxed(text: str) -> str | None:
    """Content of the last `\\boxed{...}` in `text` (braces matched), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def compute_score(completion: str, answer: str) -> float:
    box = last_boxed(completion)
    if box is None:
        return 0.0
    # math_verify's timeouts use SIGALRM, which only works on the main thread: disable them elsewhere (reward workers).
    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    config = [LatexExtractionConfig(boxed_match_priority=0)]
    try:
        gold = parse(f"\\boxed{{{answer}}}", extraction_config=config, parsing_timeout=timeout)
        pred = parse(f"\\boxed{{{box}}}", extraction_config=config, parsing_timeout=timeout)
        return float(bool(gold) and bool(pred) and verify(gold, pred, timeout_seconds=timeout))
    except Exception:
        return 0.0


if __name__ == "__main__":
    cases = [
        ("so \\boxed{42}", "42", 1.0),
        ("first \\boxed{1} then \\boxed{\\frac{1}{2}}", "0.5", 1.0),
        ("first \\boxed{1/2} then \\boxed{3}", "1/2", 0.0),
        ("\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)", 1.0),
        ("Answer: 42", "42", 0.0),
        ("\\boxed{x^2+1}", "1+x^2", 1.0),
        ("\\boxed{ 34 }", "34", 1.0),
        ("<think>\\boxed{1}</think> \\boxed{2", "2", 0.0),
    ]
    for completion, answer, want in cases:
        got = compute_score(completion, answer)
        print(f"{got} want {want}  {completion!r} vs {answer!r}")
        assert got == want
    print("reward self-test OK")
