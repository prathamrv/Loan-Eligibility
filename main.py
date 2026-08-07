"""
main.py
-------
Single entry point to run the full pipeline: load -> clean -> train ->
evaluate -> persist artifacts.

Usage:
    python main.py
"""

import json
import sys

from src.exceptions import LoanPipelineError
from src.logger import get_logger
from src.train import train_model

logger = get_logger(__name__)


def main() -> int:
    logger.info("Starting loan eligibility training pipeline")
    try:
        metrics = train_model()
    except LoanPipelineError as exc:
        logger.error("Pipeline aborted: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nTraining complete. Held-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nArtifacts saved under ./models. Launch the app with:")
    print("    streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
