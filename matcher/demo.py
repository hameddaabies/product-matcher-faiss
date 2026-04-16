"""End-to-end runnable demo.

    python -m matcher.demo

Loads retailer_a as the catalog, runs every retailer_b name as a query,
prints top matches.
"""

from __future__ import annotations

import json
from pathlib import Path

from .hybrid import HybridMatcher

MATCH_THRESHOLD = 0.70


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "fixtures"
    retailer_a = _load(root / "retailer_a.json")
    retailer_b = _load(root / "retailer_b.json")

    matcher = HybridMatcher(
        ids=[r["id"] for r in retailer_a],
        names=[r["name"] for r in retailer_a],
        alpha=0.5,
        top_k_each=10,
    )

    for row in retailer_b:
        print(f"\nQuery: {row['name']}")
        hits = matcher.match(row["name"], top_n=3)
        for h in hits:
            verdict = "MATCH" if h.score >= MATCH_THRESHOLD else "no match"
            print(f"  → {h.name:<60s} (score={h.score:.2f}) [{verdict}]")


if __name__ == "__main__":
    main()
