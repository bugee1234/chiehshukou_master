# Experiment 2 Report for Claude

## 0. Project Context

This project studies factual consistency / omission detection for biomedical lay summaries. Experiment 2 focuses on whether a filtering module is necessary and whether an automatic keep/skip strategy can safely reduce the article-level atomic fact (AF) set before omission detection.

Current pilot setting:

- Data: 10 articles total
  - PLOS: 5 articles
  - eLife: 5 articles
- Model used in current outputs: `gpt-4.1`
- Main runner: `src/experiment_2/run_exp2.py`
- Main data directory: `data/experiment_2/`
- Main results directory: `results/experiment_2/`

Important counts:

- Full article AF for pilot 10 articles: `2623`
- Part B direct-coverage ground-truth keep AF: `433`
- Part C selected keep AF: `1641`
- Part C expert summary AF: `127`

---

## 1. Experiment 2 Overview

Experiment 2 has three parts:

1. **Part A: Module 2 filtering necessity**
   - Question: Do expert lay summaries cover only a small subset of full article-level AFs?
   - Purpose: Show that many article facts are legitimate omissions / safe simplifications, so a module is needed to avoid treating every omitted article fact as an omission error.

2. **Part B: Automatic keep/skip reliability**
   - Question: Can an LLM-based module automatically identify which article AFs should be kept for downstream omission detection?
   - Purpose: Evaluate keep/skip prompts, especially whether a high-recall version can avoid dangerous skips.

3. **Part C: Cost-saving direct selected keep AF extraction**
   - Question: Can we skip full article AF extraction and directly extract only keep-worthy AFs from the article?
   - Purpose: Reduce API cost and execution time. Full AF extraction plus per-AF keep/skip is expensive because every article chunk first produces many AFs, and then every AF needs another LLM keep/skip call.

---

## 2. Part A: Module 2 Filtering Necessity

### Purpose

Part A tests whether expert lay summaries cover only a small fraction of article-level AFs.

If expert summaries cover only a small fraction, then most missing article facts are not true omissions. They are safe simplifications or reasonable exclusions. This supports the need for a filtering module before omission detection.

### Input

- Full article AFs:
  - `data/experiment_2/01_article_af/af_article.jsonl`
- Pilot subset:
  - 10 articles
  - `2623` article AFs

### Method

Part A uses a Setup C-style multiple-choice coverage evaluation.

For each article AF:

1. Generate one true option and three false/counterfactual options.
2. Add option `E. None of the above`.
3. Give the model the expert lay summary.
4. Ask the model to identify which option is supported by the lay summary.
5. If the model chooses the true option, the article AF is counted as covered.
6. If the model chooses a false option or `E`, the article AF is counted as not covered.

### Key Evaluation Logic

The relevant idea is:

```text
article AF + expert lay summary
-> Setup C-style 1T3F + None-of-the-above coverage judgement
-> covered / not covered
```

### Key Prompt Idea

The Setup C judge is strict and summary-grounded:

```text
Use ONLY the provided lay summary. Do NOT use outside knowledge.
Evaluate each option independently.
Find verbatim quotes from the lay summary.
Mark options as supported / contradicted / insufficient.
```

The prompt also allows relaxed paraphrase for true options:

```text
An option is supported if its core information is conveyed by the lay summary.
The summary may paraphrase, generalize, or abstract the option.
```

### Results

From `data/experiment_2/04_eval/ultra_recall/metrics_summary.json`:

```text
n = 2623
covered = 257
not covered = 2366
coverage_recall = 0.098
```

### Interpretation

Only about `9.8%` of article-level AFs are covered by expert summaries under the Setup C-style evaluation.

This strongly supports the need for Module 2 / filtering:

```text
If we compare every article AF against the lay summary, most absent article facts would be incorrectly treated as omissions.
```

Part A therefore supports the claim that omission detection must first distinguish expert-retained facts from safe simplifications.

---

## 3. Part B: Automatic Keep/Skip Reliability

### Purpose

Part B tests whether an LLM can automatically predict which article AFs should be kept for downstream omission detection.

The central risk is **dangerous skip**:

```text
An article AF is actually covered by the expert summary, but the module predicts skip.
```

For omission detection, dangerous skip is worse than overkeep. Overkeep increases review cost; dangerous skip removes a fact that may be needed for evaluation.

### Ground Truth

Part B uses direct coverage as the ground truth.

For each full article AF:

1. Give the article AF and the whole expert lay summary to an LLM judge.
2. Ask whether the lay summary conveys the article AF.
3. If `covered=true`, label the AF as ground-truth keep.
4. If `covered=false`, label the AF as ground-truth skip.

Direct coverage prompt path:

- `src/experiment_2/prompts/direct_coverage.txt`

Key direct coverage prompt:

```text
You are checking whether an article-level atomic fact is conveyed by an expert-written biomedical lay summary.

The lay summary may paraphrase, simplify, generalize, or abstract the article fact rather than state it verbatim.

Use a conservative but fair standard:
- Answer YES if the summary preserves the same core factual content, such as the same main entity, finding, direction, relation, outcome, or conclusion.
- Answer YES if the summary states a patient-facing generalization of the same fact.
- Answer NO if the summary only discusses the same broad topic but omits the specific claim.
- Answer NO if the summary lacks the key entity, direction, comparison, outcome, or conclusion needed to verify the fact.
- Answer NO if the summary contradicts the fact.
```

### Direct Coverage Ground Truth Results

From `data/experiment_2/04_eval/ultra_recall/metrics_summary.json`:

```text
n = 2623
ground-truth keep = 433
ground-truth skip = 2190
coverage_recall = 0.1651
```

This means the direct coverage judge says expert summaries cover about `16.51%` of article AFs.

### Keep/Skip Prediction

The keep/skip module sees only one article AF at a time and predicts whether it should be retained for lay-summary omission evaluation.

Two prompt variants were tested:

1. `balanced`
2. `ultra_recall`

### Balanced Prompt

Prompt path:

- `src/experiment_2/prompts/keep_skip_balanced.txt`

Key design:

```text
Predict whether an expert lay summary would retain this fact.
Do NOT keep every background fact.
Keep background only when it is one of the few facts needed to understand the central question or conclusion.
Choose DROP when the fact is peripheral, overly technical, too fine-grained, or likely to be compressed.
```

This prompt is more selective.

### Balanced Results

From `data/experiment_2/04_eval/balanced/confusion_matrix.csv`:

```text
overall:
tp = 137
tn = 1969
fp = 221
fn = 296
accuracy = 0.8029
precision = 0.3827
recall = 0.3164
f1 = 0.3464
dangerous_skip_rate = 0.6836
```

By source:

```text
PLOS:
recall = 0.1549
dangerous_skip_rate = 0.8451

eLife:
recall = 0.4727
dangerous_skip_rate = 0.5273
```

Interpretation:

The balanced prompt has much lower overkeep, but it is unsafe for the current purpose because dangerous skip is too high.

### Ultra Recall Prompt

Prompt path:

- `src/experiment_2/prompts/keep_skip_ultra_recall.txt`

Key design:

```text
Missing an expert-retained fact is the most dangerous error.
Over-keeping is acceptable because a human reviewer can remove extra facts later.
Prefer KEEP unless the fact is clearly unsuitable for a lay summary.
If uncertain between KEEP and DROP, choose KEEP.
```

This prompt intentionally prioritizes recall over precision.

### Ultra Recall Results

From `data/experiment_2/04_eval/ultra_recall/confusion_matrix.csv`:

```text
overall:
tp = 422
tn = 698
fp = 1492
fn = 11
accuracy = 0.4270
precision = 0.2205
recall = 0.9746
f1 = 0.3596
dangerous_skip_rate = 0.0254
```

By source:

```text
PLOS:
tp = 205
tn = 280
fp = 492
fn = 8
recall = 0.9624
dangerous_skip_rate = 0.0376

eLife:
tp = 217
tn = 418
fp = 1000
fn = 3
recall = 0.9864
dangerous_skip_rate = 0.0136
```

### Part B Evaluation

Part B uses a standard binary classification evaluation:

```text
ground truth = direct coverage result
prediction = module keep/skip result
```

Confusion matrix definitions:

```text
TP = ground-truth keep, module keep
TN = ground-truth skip, module skip
FP = ground-truth skip, module keep
FN = ground-truth keep, module skip
```

Critical metric:

```text
dangerous_skip_rate = FN / (TP + FN)
```

### Part B Interpretation

The ultra_recall prompt is much better aligned with the safety goal.

It keeps `422 / 433` ground-truth keep facts and misses only `11`.

The tradeoff is severe overkeep:

```text
fp = 1492
precision = 0.2205
```

Recommended interpretation:

```text
The ultra_recall module is not a precise automatic deletion tool.
It is a conservative high-recall pre-filter designed to minimize dangerous skips.
It is suitable for reducing catastrophic omission of expert-retained facts, but still requires downstream filtering or human review due to overkeep.
```

---

## 4. Part C: Direct Selected Keep AF Extraction

### Purpose

Part C was introduced because the original full pipeline is expensive and slow:

```text
article -> full article AF extraction -> per-AF keep/skip judgement
```

This requires:

1. API calls to extract many full article AFs.
2. Additional API calls for keep/skip classification for each AF.
3. More execution time because full articles generate many facts.

The cost-saving idea is:

```text
article -> directly extract selected keep AF
```

Instead of extracting every article AF and later filtering, Part C asks the model to directly output only AFs that might reasonably be kept for an expert lay summary.

### Current Part C Pipeline

Part C has three stages:

1. Extract summary AF from expert lay summaries.
2. Directly extract selected keep AF from articles.
3. Evaluate whether selected keep AFs cover expert summary AFs.

### Stage 1: Summary AF Extraction

Command:

```bash
python3 src/experiment_2/run_exp2.py partc_extract_summary_af --model gpt-4.1 --pilot-per-source 5
```

Output:

- `data/experiment_2/part_c_selective_keep_af/01_summary_af/summary_af.jsonl`
- `data/experiment_2/part_c_selective_keep_af/01_summary_af/summary_af_metadata.json`

Prompt path:

- `src/experiment_2/prompts/partc_summary_af_extraction.txt`

Key prompt:

```text
You are an expert at extracting atomic facts from expert-written biomedical lay summaries.

An Atomic Fact is a complete sentence that:
1. Contains exactly ONE fact, finding, claim, or concrete piece of information.
2. Is semantically independent and understandable without surrounding context.
3. Resolves pronouns and vague references when possible.
4. Includes necessary qualifiers.
5. Never introduces information not present in the source text.
```

Result:

```text
summary_af_total = 127
PLOS = 45
eLife = 82
```

### Stage 2: Direct Selected Keep AF Extraction

Command:

```bash
python3 src/experiment_2/run_exp2.py partc_extract_keep_af --model gpt-4.1 --chunk-words 1200 --overlap-words 120 --pilot-per-source 5
```

Output:

- `data/experiment_2/part_c_selective_keep_af/02_selected_keep_af/selected_keep_af.jsonl`
- `data/experiment_2/part_c_selective_keep_af/02_selected_keep_af/selected_keep_af_metadata.json`

Prompt path:

- `src/experiment_2/prompts/selective_keep_af_ultra_recall.txt`

Key prompt design:

```text
You are extracting KEEP atomic facts from a biomedical research article for an expert-written lay summary.

Your task is NOT to extract every article fact.
Extract facts that an expert-written lay summary might retain, paraphrase, simplify, or use as explanatory scaffolding.

Missing an expert-retained fact is the most dangerous error.
Over-extracting is acceptable when the fact is plausibly useful for a lay summary.
```

The current prompt explicitly includes eLife-style background rescue:

```text
Expert lay summaries, especially eLife-style summaries, often include general educational background before describing the study itself.
Do not drop a fact merely because it is not novel.

KEEP:
- core findings
- study motivation
- high-level method purpose
- background needed for lay understanding
- definitions of key concepts
- analogies
- general bridge facts
- normal biological processes before explaining abnormal/disease process

Do NOT extract:
- routine protocol steps
- reagent catalog details
- software settings
- statistical procedures
- implementation details
- overly fine-grained parameters
```

Current extraction result:

```text
selected_keep_af_total = 1641
PLOS = 614
eLife = 1027
failed_chunks = 0
```

By article:

```text
exp2_plos_001 = 121
exp2_plos_002 = 123
exp2_plos_003 = 111
exp2_plos_004 = 149
exp2_plos_005 = 110
exp2_elife_001 = 150
exp2_elife_002 = 95
exp2_elife_003 = 302
exp2_elife_004 = 207
exp2_elife_005 = 273
```

Cost/time note:

This run still used many tokens:

```text
input tokens = 237800
output tokens = 187163
total tokens = 424963
```

So Part C reduces the need for a separate per-AF keep/skip stage, but the direct extraction itself can still be expensive, especially when the prompt is high-recall.

### Stage 3: Summary-Facing Recall Evaluation

Command:

```bash
python3 src/experiment_2/run_exp2.py partc_eval_selected_keep_af --model gpt-4.1 --pilot-per-source 5
```

Output:

- `data/experiment_2/part_c_selective_keep_af/03_eval/summary_af_coverage_by_selected_keep.jsonl`
- `data/experiment_2/part_c_selective_keep_af/03_eval/partc_metrics_summary.json`
- `data/experiment_2/part_c_selective_keep_af/03_eval/partc_summary.csv`

Prompt path:

- `src/experiment_2/prompts/partc_selected_keep_covers_summary_af.txt`

Key prompt:

```text
You are checking whether an expert-summary atomic fact is covered by a list of selected keep-worthy article facts.

Answer YES if at least one selected article fact conveys the same core information as the expert-summary fact.

Use a fair but conservative standard:
- YES if the same core entity, relation, finding, outcome, mechanism, or conclusion is preserved.
- YES if a selected article fact is a more detailed version of the summary fact.
- NO if selected facts only discuss the same broad topic but omit the key claim.
- NO if selected facts contradict the summary fact.
```

### Part C Evaluation Method

Part C currently evaluates recall in a summary-facing way:

```text
For each expert summary AF:
    Take all Part C selected keep AFs from the same article.
    Ask whether any selected keep AF covers the summary AF.
```

Metric:

```text
summary_fact_recall = covered summary AF / all summary AF
```

This is not a full confusion matrix because all 127 summary AFs are positive/relevant items. It measures whether Part C missed facts that appear in expert summaries.

### Part C Results

From `data/experiment_2/part_c_selective_keep_af/03_eval/partc_summary.csv`:

```text
summary_af_total = 127
selected_keep_af_total = 1641
summary_fact_recall = 0.8898
covered_summary_af_count = 113
missed_summary_af_count = 14
```

By source:

```text
PLOS:
summary_af_total = 45
covered = 45
summary_fact_recall = 1.0000

eLife:
summary_af_total = 82
covered = 68
summary_fact_recall = 0.8293
```

By article:

```text
exp2_plos_001: 10/10 = 1.0000
exp2_plos_002: 9/9 = 1.0000
exp2_plos_003: 8/8 = 1.0000
exp2_plos_004: 9/9 = 1.0000
exp2_plos_005: 9/9 = 1.0000

exp2_elife_001: 15/17 = 0.8824
exp2_elife_002: 12/16 = 0.7500
exp2_elife_003: 20/23 = 0.8696
exp2_elife_004: 14/19 = 0.7368
exp2_elife_005: 7/7 = 1.0000
```

### Part C Interpretation

Part C successfully covers all PLOS summary AFs in the pilot, but eLife remains weaker.

The current Part C result:

```text
1641 selected keep AFs
113 / 127 summary AFs covered
summary recall = 0.8898
```

This is useful but does not yet match the safety level of Part B ultra_recall.

Key limitation:

Part C direct extraction can still miss eLife-style educational background, definitions, and explanatory bridge facts, even after strengthening the prompt.

Part C should therefore currently be interpreted as:

```text
A cost-saving exploratory variant that reduces the need for a separate full AF + keep/skip pipeline, but whose recall is not yet high enough to replace the ultra_recall keep/skip pipeline as the main safety-oriented result.
```

---

## 5. Cross-Part Comparison

### Main Result Table

| Experiment | Unit / Denominator | Main Metric | Result | Interpretation |
|---|---:|---:|---:|---|
| Part A Setup C coverage | 2623 full article AFs | coverage recall | 0.098 | Expert summaries cover only a small fraction of article AFs; Module 2/filtering is necessary. |
| Part B direct coverage GT | 2623 full article AFs | ground-truth keep rate | 0.1651 | Direct coverage judge identifies 433 / 2623 article AFs as summary-covered. |
| Part B balanced keep/skip | 433 GT keep AFs | recall | 0.3164 | Too many dangerous skips; unsafe. |
| Part B ultra_recall keep/skip | 433 GT keep AFs | recall | 0.9746 | Very low dangerous skip; best safety-oriented filter. |
| Part C summary-facing eval | 127 summary AFs | summary_fact_recall | 0.8898 | Direct extraction is promising but misses 14 summary facts, mostly from eLife. |

### Important Distinction Between Part B and Part C Evaluation

Part B evaluates binary classification over the full article AF universe:

```text
full article AF -> direct coverage GT -> module keep/skip prediction
```

Part C currently evaluates summary-facing recall:

```text
expert summary AF -> covered by selected keep AF?
```

So Part B and Part C are not exactly the same evaluation denominator.

Part B is best for measuring keep/skip dangerous skip under a fixed full AF universe.

Part C is best for measuring whether direct selected keep extraction covers the factual content of expert summaries.

---

## 6. Suggested Narrative for Thesis

### Part A Narrative

Expert summaries cover only a small subset of article-level AFs. Therefore, many omissions are safe simplifications. This motivates an explicit filtering stage before omission detection.

### Part B Narrative

Automatic keep/skip filtering is possible, but prompt design matters.

The balanced prompt is too aggressive and causes many dangerous skips.

The ultra_recall prompt keeps nearly all expert-summary-covered article AFs:

```text
recall = 0.9746
dangerous skip rate = 0.0254
```

Its weakness is overkeep:

```text
precision = 0.2205
overkeep count = 1492
```

Thus ultra_recall should be framed as a conservative high-recall pre-filter, not a final automatic pruning tool.

### Part C Narrative

Part C was motivated by API cost and execution time. The full pipeline requires extracting all AFs and then judging keep/skip per AF.

Part C attempts to reduce cost by directly extracting selected keep AFs from articles.

The current result:

```text
selected keep AF = 1641
summary AF recall = 0.8898
```

This is promising but not yet strong enough to replace Part B ultra_recall as the main safety-oriented method.

The main issue is eLife:

```text
PLOS summary recall = 1.0000
eLife summary recall = 0.8293
```

This suggests eLife expert summaries include more explanatory background, definitions, and analogies that are difficult to recover with direct article-to-keep extraction.

---

## 7. Key File Map

### Code

```text
src/experiment_2/run_exp2.py
```

### Prompts

```text
src/experiment_2/prompts/direct_coverage.txt
src/experiment_2/prompts/keep_skip_balanced.txt
src/experiment_2/prompts/keep_skip_ultra_recall.txt
src/experiment_2/prompts/selective_keep_af_ultra_recall.txt
src/experiment_2/prompts/partc_summary_af_extraction.txt
src/experiment_2/prompts/partc_selected_keep_covers_summary_af.txt
```

### Part A / B Outputs

```text
data/experiment_2/01_article_af/af_article.jsonl
data/experiment_2/02_coverage_setup_c/coverage_predictions.jsonl
data/experiment_2/02b_direct_coverage/coverage_direct_predictions.jsonl
data/experiment_2/03_keep_skip/balanced/keep_skip_predictions.jsonl
data/experiment_2/03_keep_skip/ultra_recall/keep_skip_predictions.jsonl
data/experiment_2/04_eval/balanced/confusion_matrix.csv
data/experiment_2/04_eval/balanced/metrics_summary.json
data/experiment_2/04_eval/ultra_recall/confusion_matrix.csv
data/experiment_2/04_eval/ultra_recall/metrics_summary.json
```

### Part C Outputs

```text
data/experiment_2/part_c_selective_keep_af/01_summary_af/summary_af.jsonl
data/experiment_2/part_c_selective_keep_af/02_selected_keep_af/selected_keep_af.jsonl
data/experiment_2/part_c_selective_keep_af/03_eval/summary_af_coverage_by_selected_keep.jsonl
data/experiment_2/part_c_selective_keep_af/03_eval/partc_summary.csv
data/experiment_2/part_c_selective_keep_af/03_eval/partc_metrics_summary.json
```

---

## 8. Current Bottom Line

Part A shows Module 2/filtering is necessary because expert summaries cover only a small fraction of full article AFs.

Part B shows the ultra_recall keep/skip prompt is the best current safety-oriented method:

```text
recall = 0.9746
dangerous skip rate = 0.0254
```

Part C was motivated by API cost and runtime. It directly extracts selected keep AFs and reaches:

```text
selected keep AF = 1641
summary fact recall = 0.8898
```

Part C is promising but not yet strong enough to replace the Part B ultra_recall pipeline, especially for eLife-style summaries.

