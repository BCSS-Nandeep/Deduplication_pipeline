"""python -m dedup_pipeline.face_recognition health   (read-only liveness check of FR-DOPAMS)"""

import argparse
import sys

from .client import FrDopamsClient
from .config import FrConfig
from .errors import FrError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dedup_pipeline.face_recognition")
    parser.add_argument("command", choices=["health"])
    parser.parse_args(argv)
    try:
        client = FrDopamsClient(FrConfig.from_env())
        client.verify_service()
    except FrError as exc:
        print(f"FR-DOPAMS NOT OK ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print("FR-DOPAMS OK: /health is exactly status=ok and /openapi.json identifies the face service with POST /compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
