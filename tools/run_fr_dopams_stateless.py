"""Run the real FR-DOPAMS app for testing POST /compare WITHOUT touching any database.

FR-DOPAMS normally runs ``db.init_db()`` at startup (CREATE EXTENSION + create_all). That is a schema
change, so this launcher replaces ``initialize_system`` with a no-op, and refuses to start unless
DATABASE_URL is the dead sentinel below: any accidental database access then fails loudly instead of
reaching a real server. Everything else is the real application: its routes, middleware, model and code.

    $env:DATABASE_URL = "postgresql://nodb:nodb@127.0.0.1:1/nodb"
    $env:COMPARE_API_KEY = "<at least 16 characters>"
    python tools/run_fr_dopams_stateless.py 8010
"""

import os
import sys
from pathlib import Path

FR_DIR = Path(os.environ.get("FR_DOPAMS_SOURCE_DIR") or Path(__file__).resolve().parents[2] / "FR-DOPAMS")
SENTINEL = "postgresql://nodb:nodb@127.0.0.1:1/nodb"


def main() -> int:
    if os.environ.get("DATABASE_URL") != SENTINEL:
        print(f"Refusing to start: set DATABASE_URL to the dead sentinel {SENTINEL} (no real database may be reachable).")
        return 2
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
    sys.path.insert(0, str(FR_DIR))
    import app as fr_app  # the real FR-DOPAMS application module
    import uvicorn

    fr_app.initialize_system = lambda: None  # skip db.init_db(): no extension, no tables, no connection
    uvicorn.run(fr_app.app, host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
