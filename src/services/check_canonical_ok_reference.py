"""Read-only ODIN canonical OK archive reference verification.

Run: python -m src.services.check_canonical_ok_reference
"""
from __future__ import annotations

import json
from src.core.operator_confirmed_archive_labels import canonical_ok_source_status


def main() -> int:
    report = canonical_ok_source_status()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
