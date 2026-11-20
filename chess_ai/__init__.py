"""
chess_ai
Gói mã nguồn trung tâm của Hệ thống Game AI Cờ vua Đa cấp độ (8x8 FIDE).
"""

__version__ = "0.1.0"


import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
