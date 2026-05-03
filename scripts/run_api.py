from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from src.api.settings import SETTINGS

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=SETTINGS.host, port=SETTINGS.port, reload=SETTINGS.reload)
