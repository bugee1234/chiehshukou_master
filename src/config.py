from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

MODEL_GENERATOR = "gpt-4.1"
MODEL_GOLD_SOLVER = "gpt-4.1"
MODEL_BLIND_SOLVERS = ["gpt-4.1", "gpt-4o-mini"]
# Backward compatibility for older scripts that still reference MODEL_SOLVER.
MODEL_SOLVER = MODEL_GOLD_SOLVER

EXPERIMENT_NAME = "experiment_0"
SEED = 42


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
PROMPTS_DIR = ROOT_DIR / "prompts"
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_ATOMIC_FACTS_DIR = DATA_DIR / "atomic_facts"
DATA_QUESTIONS_DIR = DATA_DIR / "questions"
DATA_SOLVER_RESULTS_DIR = DATA_DIR / "solver_results"
RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENT_DIR = RESULTS_DIR / EXPERIMENT_NAME
LOGS_DIR = ROOT_DIR / "logs"


for path in [
    PROMPTS_DIR,
    DATA_DIR,
    DATA_RAW_DIR,
    DATA_ATOMIC_FACTS_DIR,
    DATA_QUESTIONS_DIR,
    DATA_SOLVER_RESULTS_DIR,
    RESULTS_DIR,
    EXPERIMENT_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
