from src.utils.config import load_config

from src.orchestration.orchestrator import Orchestrator


if __name__ == "__main__":
    config = load_config("experiments_config.json")
    orchestrator = Orchestrator(config)
    orchestrator.run()
