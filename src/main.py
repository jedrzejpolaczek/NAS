from utils.config import load_config
from utils.logger import get_logger
from orchestration.orchestrator import Orchestrator


if __name__ == "__main__":
    config = load_config("experiments_config.json")
    nas_logger = get_logger("nas logger", config["log_dir"])
    orchestrator = Orchestrator(config)
    orchestrator.run()
