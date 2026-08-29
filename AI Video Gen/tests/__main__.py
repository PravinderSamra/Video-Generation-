"""Run every test suite: `python -m tests`"""

from __future__ import annotations

import sys

from . import test_pipeline, test_review

SUITES = [("pipeline", test_pipeline), ("review", test_review)]


def main() -> int:
    failed = []
    for name, suite in SUITES:
        print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
        if suite.main() != 0:
            failed.append(name)
    print(f"\n{'=' * 62}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
