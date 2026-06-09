#!/usr/bin/env python
"""vrag CLI 진입점.  사용: python scripts/vrag.py {list|build|infer} ..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_rag_cxr.cli import main
if __name__ == "__main__":
    main()
