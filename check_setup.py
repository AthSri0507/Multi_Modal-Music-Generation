#!/usr/bin/env python3
"""Basic environment checks for this workspace."""

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def check_python() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 8)
    print(f"[{'OK' if ok else 'FAIL'}] Python version: {v.major}.{v.minor}.{v.micro} (need >= 3.8)")
    return ok


def check_dirs() -> bool:
    required = [
        ROOT / "src",
        ROOT / "src" / "config",
        ROOT / "src" / "data",
        ROOT / "data",
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
    ]
    ok = True
    for d in required:
        exists = d.exists() and d.is_dir()
        print(f"[{'OK' if exists else 'FAIL'}] Directory: {d.relative_to(ROOT)}")
        ok = ok and exists
    return ok


def check_packages() -> bool:
    pkgs = ["pandas", "numpy", "sklearn", "torch", "transformers", "kagglehub"]
    ok = True
    for p in pkgs:
        try:
            __import__(p)
            print(f"[OK] Package: {p}")
        except Exception:
            print(f"[FAIL] Package: {p}")
            ok = False
    return ok


def check_disk() -> bool:
    _, _, free = shutil.disk_usage(str(ROOT))
    free_gb = free / (1024**3)
    ok = free_gb >= 5
    print(f"[{'OK' if ok else 'FAIL'}] Free disk: {free_gb:.2f} GB (need >= 5 GB)")
    return ok


def check_requirements_file() -> bool:
    req = ROOT / "requirements.txt"
    ok = req.exists()
    print(f"[{'OK' if ok else 'FAIL'}] requirements.txt")
    return ok


def main() -> int:
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    checks = [
        check_python,
        check_dirs,
        check_packages,
        check_disk,
        check_requirements_file,
    ]

    results = [c() for c in checks]
    passed = sum(1 for r in results if r)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
