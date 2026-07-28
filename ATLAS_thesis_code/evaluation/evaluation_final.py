import os, sys, json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "AlignScore" / "src"))
sys.path.append(str(ROOT_DIR / "summac"))
import torch
import nltk 
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
from alignscore import AlignScore
from lens import download_model, LENS
from summac.model_summac import SummaCConv
import argparse
from huggingface_hub import hf_hub_download
import evaluate

def calc_rouge(preds, refs):
  # Get ROUGE F1 scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  return (np.mean([s['rouge1'].fmeasure for s in scores])+ \
         np.mean([s['rouge2'].fmeasure for s in scores])+ \
         np.mean([s['rougeLsum'].fmeasure for s in scores]))/3.0

def cal_bleu(preds, refs):
  bleu = evaluate.load('sacrebleu')
  scores = bleu.compute(predictions=preds, references=refs,tokenize="13a")["score"]
  return scores

def cal_meteor(preds, refs):
  meteor = evaluate.load('meteor')
  scores = meteor.compute(predictions=preds, references=refs)["meteor"]
  return scores

def calc_bertscore(preds, refs):
  # Get BERTScore F1 scores
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda')
  return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)

def calc_lens(preds, refs, docs):
  lens_path = download_model("davidheineman/lens")
  metric = LENS(lens_path, rescale=True)
  abstracts = [d.split("\n")[0] for d in docs]
  refs = [[x] for x in refs]

  scores = metric.score(abstracts, preds, refs, batch_size=8)#, gpus=1
  return np.mean(scores)

def calc_alignscore(preds, docs):
  model_path = hf_hub_download(repo_id="yzha/AlignScore", filename="AlignScore-base.ckpt")
  alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda', \
                           ckpt_path=model_path, evaluation_mode='nli_sp')
  return np.mean(alignscorer.score(contexts=docs, claims=preds))

def cal_summac(preds, docs):
  start_file = ROOT_DIR / "summac" / "summac_conv_vitc_sent_perc_e.bin"
  model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file=str(start_file), agg="mean")
  return np.mean(model_conv.score(docs, preds)['scores'])

def evaluate_all(preds,refs_dicts,task_name,summac_docs=None):
  # Load data from files
  # refs_dicts = read_file_lines(gold_path)
  # preds = read_file_lines(pred_path)

  assert len(refs_dicts)==len(preds)
  refs = [d['reference'] for d in refs_dicts]
  if task_name == "lay_summ":
    docs = [d['document'] for d in refs_dicts]
    if summac_docs is None:
      summac_docs = docs
    assert len(summac_docs) == len(preds)
  
  score_dict = {}

  # Relevance scores
  print("[evaluation 1/10] ROUGE", flush=True)
  score_dict['ROUGE'] = calc_rouge(preds, refs)
  print("[evaluation 2/10] BLEU", flush=True)
  score_dict['BLEU'] = cal_bleu(preds, refs)
  print("[evaluation 3/10] METEOR", flush=True)
  score_dict['METEOR'] = cal_meteor(preds, refs)  
  print("[evaluation 4/10] BERTScore", flush=True)
  score_dict['BERTScore'] = calc_bertscore(preds, refs)
  

  # # Readability scores
  print("[evaluation 5-7/10] FKGL, DCRS, CLI", flush=True)
  fkgl_score, cli_score, dcrs_score = calc_readability(preds)
  score_dict['FKGL'] = fkgl_score
  score_dict['DCRS'] = dcrs_score
  score_dict['CLI'] = cli_score

  # Factuality scores
  if task_name == "lay_summ":
    print("[evaluation 8/10] LENS", flush=True)
    score_dict['LENS'] = calc_lens(preds, refs, docs)
    print("[evaluation 9/10] AlignScore", flush=True)
    score_dict['AlignScore'] = calc_alignscore(preds, docs)   
    print("[evaluation 10/10] SummaC", flush=True)
    score_dict['SummaC'] = cal_summac(preds, summac_docs)

  print(score_dict)

  return score_dict
