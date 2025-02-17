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
        default="experiments/configs/all_algorithms.json",
        help="Path to the experiments configuration file. \
            Default ./experiments/configs/all_algorithms.json"
    )
    args = parser.parse_args()

    try:
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        print("Loading configuration...")
        config = load_config(args.config)

        print("Set logger...")
        logger = get_logger(
            name="NAS",
            log_file=config["log_dir"]
        )

    except Exception as e:
        print("Orchestration failed due to problem with setting logger: %s", e)
        sys.exit(1)
    
    try:        
        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator(config, logger)

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
