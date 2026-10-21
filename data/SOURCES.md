# Published case sources

The 4,673 inputs built from published datasets are not redistributed. `inputs_public.csv.gz` lists, for each input, the source record, our edits to its text (`[start, end, text]` spans on the source text; the added text is ours) and the SHA-256 of the final input. `src/build_public_inputs.py` rebuilds the inputs from the files below and checks every hash.

| Dataset | Used here | Where to get it | Licence |
|---|---|---|---|
| AcuityBench (Linzmayer et al., arXiv:2605.11398) | 802 presentations: HealthBench 257, PMR-Reddit 362, PMR-Synth 60, structured triage 78, Semigran 45. 300 of these, each with seven versions (2,100 inputs) | Authors' release, https://anonymous.4open.science/r/acuity-bench-C9CD. Build from source with `python build_benchmark.py`, then use `data/03_transformed/acuitybench_transformed.csv`. The script reads `dataset`, `source_id` and the single user message in `conversational_prompt` | Not yet stated by the authors; component datasets keep their own licences |
| Ramaswamy et al., Nat Med 2026 | 78 cases in 16 versions (1,248 inputs), with the answer-format instructions removed | https://github.com/ashwinra-code/gpt-health-eval, tag `v1.0`, `data/DataExpanded_FINAL.csv` | MIT |
| Levine et al., Lancet Digit Health 2024 | 48 vignettes (eTable 2) | medRxiv, doi:10.1101/2023.01.30.23285067, supplementary file `media-1.docx` | CC BY-NC-ND 4.0 |
| Sen2Med (Gourabathina et al.) | 65 AskDocs baseline messages, each with seven versions (455 inputs) | https://github.com/abinithago/Sen2Med, commit `46bee75f869a1fbe96966b66180f065d5bd12526`, `dataset.csv` | Data CC BY-SA 4.0; code MIT |
| HELPMed (Bean et al., Nat Med 2026) | 10 scenarios and their third-person versions (20 inputs) | https://github.com/am-bean/HELPMed, commit `35b50de8fa5debd1f1ebc2c768496dd3b6ad4cf4`, `data/main/scenarios.jsonl` and `data/scenarios_3p.jsonl` | MIT |

```
python src/build_public_inputs.py --acuitybench acuitybench_transformed.csv --ramaswamy DataExpanded_FINAL.csv \
    --levine media-1.docx --sen2med dataset.csv --helpmed HELPMed/ --out inputs_public.jsonl
```

`--helpmed` is a clone of the HELPMed repository at the commit above. Any source can be left out. A hash mismatch means the source file differs from the version used here; the SHA-256 of each file we read:

| File | SHA-256 |
|---|---|
| `acuitybench_transformed.csv` | `47010d1c7a27f97c4c5100929715e9a56d0cb5d6cde82f0e8441821defac5956` |
| `DataExpanded_FINAL.csv` | `31861575a5eaa7b9cc001b4c90af814dc09fc4fcdec453022a2fdb8773ffa741` |
| `media-1.docx` | `880251ffc6bdcca78393f1ce7adcbeb288d52ea5b324e9f0d864191114beb535` |
| `dataset.csv` (Sen2Med) | `381cc0296b525bf67ec52860caedf3c171bf94e41d1c993498e816b154e2e1fb` |
| `scenarios.jsonl` (HELPMed) | `9214840c4fccb03473c0b6e2ad130b84830b606494756207081284c03ee4f401` |
| `scenarios_3p.jsonl` (HELPMed) | `415a4cfd557e75b7b3e636a27b83eeabe7022e20ea6e085822bc7baca0b9232d` |
