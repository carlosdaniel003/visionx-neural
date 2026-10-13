"""Probe real de console Windows: exige UTF-8 no subprocesso de manutenção."""
from __future__ import annotations

import sys

if __name__ == "__main__":
    print("✅ Tesseract encontrado: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("ENCODING:", sys.stdout.encoding)
    if sys.stdout.encoding.lower().replace("-", "") != "utf8":
        raise RuntimeError("Subprocesso iniciou sem saída UTF-8")
