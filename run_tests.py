#!/usr/bin/env python
"""隔离模式测试入口。

`python -I` 不把当前工作目录加入 sys.path，本入口在导入被测包之前
显式加入仓库根目录，使测试可在 -I（忽略环境变量、不加载 site 包）下运行。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
