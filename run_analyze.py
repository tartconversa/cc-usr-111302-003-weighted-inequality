#!/usr/bin/env python
"""隔离模式 CLI 入口：显式注入仓库根目录后转交分析器 main。

用法：python -I run_analyze.py <data_path> <config_path> [--out DIR] [--replicates N] [--seed S]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inequality_analysis import main  # noqa: E402

if __name__ == "__main__":
    main()
