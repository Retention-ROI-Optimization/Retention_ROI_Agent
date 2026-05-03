from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    port = int(os.getenv("RETENTION_API_PORT", "8010"))
    host = os.getenv("RETENTION_API_HOST", "0.0.0.0")
    uvicorn.run("src.autoops_universal.api_app:app", host=host, port=port, reload=False)
