import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.orchestration.orchestrator import Orchestrator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments orchestration."
    )
    parser.add_argument(
        "--config",
        default="experiments_config.json",
        help="Path to the experiments configuration file. \
            Default ./experiments_config.json"
    )
    args = parser.parse_args()

    logger = get_logger(
        name="NAS",
        log_file=args.config["log_dir"]
    )

    try:
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        logger.info("Loading configuration...")
        config = load_config(args.config)

        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator(config)

        logger.info("Starting orchestration...")
        orchestrator.run()

        logger.info("Orchestration completed successfully")

    except FileNotFoundError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    except Exception as e:
        logger.error("Orchestration failed: %s", e)
        sys.exit(1)
