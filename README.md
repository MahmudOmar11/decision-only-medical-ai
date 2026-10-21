# Decision-only medical AI: consistency and pairing with a language model

Code and data for "Decision-only medical AI: early evidence and pairing with language models" (Omar et al.).

Two decision models (Jev and OpenAI Decisions), a large language model (GPT-6 Luna) and two teams, each pairing a decision model with the LLM, answered the same 2,134 starting case presentations (25,158 case-and-question entries). The main outcome is excess decision change: how many more recommendations changed after an added social label or pressure cue than after a neutral addition.

## Layout

```
data/
  inputs_controlled.jsonl.gz  20,485 constructed inputs: id, set, family, question, text
  inputs_public.csv.gz        4,673 inputs from published datasets, stored as edits to the source text
  SOURCES.md                  where to get the published datasets and how to rebuild those inputs
  triage_question.json        the triage question used by every published-source input
  thresholds.json             frozen workflow thresholds by team, case set, question and original answer level
src/
  models.py                   requests to Jev, OpenAI Decisions and the LLM, and the decision rule
  workflow.py                 the paired workflow
  run.py                      runs one system over an input file
  build_public_inputs.py      rebuilds the published-source inputs and checks each against its SHA-256
  analyze.py                  recomputes the reported estimates from the decisions
  figures.py                  Figures 1 and 2
results/
  decisions.csv.gz            each system's decision for every entry (empty when unavailable)
  comparisons.csv.gz          the original, altered and neutral entries that form each comparison
  triage_agreement.csv        agreement with the reference range for the 597 AcuityBench presentations
  source_scenarios.csv        case formats that share a source scenario (AcuityBench, Ramaswamy)
  reported/                   Supplementary Data 1 to 19, as reported
```

## Reproduce the analyses (no model calls)

```
pip install -r requirements.txt
python src/analyze.py
python src/figures.py
```

This writes `results/analysis/` and `results/figures/` in a few seconds. The outputs reproduce every paired decision-change row of Supplementary Data 1 and 3, Supplementary Data 15 to 19, and Figures 1 and 2.

## Run the models

```
export OPENROUTER_API_KEY=...   # Jev and the LLM
export OPENAI_API_KEY=...       # OpenAI Decisions
python src/run.py jev data/inputs_controlled.jsonl.gz runs/jev.jsonl
```

Systems are `jev`, `decisions`, `llm`, `jev_team` and `decisions_team`. `--dry-run` prints the request for the first input without calling anything, and `--limit N` runs the first N inputs. Runs resume from the output file. For the published-source inputs, first build them as described in `data/SOURCES.md`.

Settings as in the study:
- Jev: `typesafe/jev-1.13` (snapshot 1.13-20260917) through OpenRouter, one call per input.
- OpenAI Decisions: `gpt-6-luna`, three calls per input, probabilities averaged; refusals are missing.
- LLM: GPT-6 Luna at low reasoning effort with a 2,000-token limit, three calls per input, majority answer (ties to the earlier option). The Decisions team called the same model directly through OpenAI: set `LLM_PROVIDER=openai`.
- The paired workflow is described at the top of `src/workflow.py`.

The models are proprietary and can change without notice, so new runs may differ from the recorded decisions.

## Data notes

Constructed cases were generated with Claude Opus 5.5 and screened by two physicians. In entry ids, `placebo` marks a neutral addition and `base` the original text. The published datasets are not redistributed; `data/SOURCES.md` gives their sources and licences.

## License

MIT (see `LICENSE`). The published datasets keep their own licences.
