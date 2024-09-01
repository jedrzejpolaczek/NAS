from utils.config import load_config
from orchestration.orchestrator import Orchestrator


if __name__ == "__main__":
    config = load_config('config.json')
    orchestrator = Orchestrator(config)
    orchestrator.run()
