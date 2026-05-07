# Detecting Information Errors and Omissions in Medical Lay Summaries

This repository contains experiment code for automated factual consistency evaluation of biomedical lay summaries.

## Setup

1. Clone this repository.
2. Create a virtual environment:
   - macOS/Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create environment file:
   - macOS/Linux:
     ```bash
     cp .env.example .env
     ```
   - Windows (PowerShell):
     ```powershell
     Copy-Item .env.example .env
     ```
5. Fill your API key in `.env`.

## Folder Structure

- `src/`: core Python modules (config, utilities, clients).
- `prompts/`: prompt templates and prompt-related assets.
- `scripts/`: runnable scripts for setup, preprocessing, and experiments.
- `data/raw/`: raw input data files.
- `data/atomic_facts/`: extracted atomic fact files.
- `data/questions/`: generated QA files for verification.
- `data/solver_results/`: model solver outputs.
- `results/experiment_0/`: experiment outputs and summaries.
- `logs/`: runtime logs.

## Progress

- [x] Step 1: Project setup
- [x] Step 2
- [x] Step 3
- [x] Step 4
- [x] Step 5
- [x] Step 6
- [x] Step 7

Final test bank: see `results/experiment_0/final_test_bank.jsonl` (325 questions)  
Full report: `results/experiment_0/report.md`
