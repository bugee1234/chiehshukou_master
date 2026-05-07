from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src  # noqa: F401
from src import config
from src.utils import OpenAIClient, setup_logger


def main() -> None:
    logger = setup_logger("smoke_test", str(config.LOGS_DIR / "smoke_test.log"))
    logger.info("Smoke test started.")

    print("=== Import check ===")
    print(f"src package loaded: {src.__name__}")
    print()

    print("=== Config paths ===")
    print(f"ROOT_DIR: {config.ROOT_DIR}")
    print(f"PROMPTS_DIR: {config.PROMPTS_DIR}")
    print(f"DATA_RAW_DIR: {config.DATA_RAW_DIR}")
    print(f"DATA_ATOMIC_FACTS_DIR: {config.DATA_ATOMIC_FACTS_DIR}")
    print(f"DATA_QUESTIONS_DIR: {config.DATA_QUESTIONS_DIR}")
    print(f"DATA_SOLVER_RESULTS_DIR: {config.DATA_SOLVER_RESULTS_DIR}")
    print(f"EXPERIMENT_DIR: {config.EXPERIMENT_DIR}")
    print(f"LOGS_DIR: {config.LOGS_DIR}")
    print()

    client = OpenAIClient(logger=logger)
    response = client.chat(
        model=config.MODEL_SOLVER,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with the single word: pong"},
        ],
        temperature=0.0,
    )

    print("=== OpenAI response ===")
    print(f"content: {response['content']}")
    print(f"usage: {response['usage']}")
    print(f"total_usage: {OpenAIClient.get_total_usage()}")
    logger.info("Smoke test completed.")


if __name__ == "__main__":
    main()
