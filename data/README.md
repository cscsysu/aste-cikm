# Data

The ASTE benchmarks (Rest14, Lap14, Rest15, Rest16) are not redistributed
here.  Obtain the raw data from the original sources and convert it to the
JSONL schema expected by the training code (see "Format" below).

## Raw sources

- **SemEval Triplet data** (Xu et al., 2020): public release with the
  unified ASTE annotations on top of SemEval-2014 Task 4 (Laptop +
  Restaurant), SemEval-2015 Task 12, and SemEval-2016 Task 5.

After downloading the raw triplet files (`train.txt` / `dev.txt` /
`test.txt` per dataset), run our conversion utility:

```bash
python scripts/convert_data.py --raw_dir <raw_dir> --out_dir data/aste
```

(`scripts/convert_data.py` is not shipped — write a few-line conversion
matching the format below; alternatively the raw SemEval Triplet
distribution already comes in a near-identical format.)

## Format

Each split is a JSONL file with one example per line:

```json
{
  "id": "rest14_train_0",
  "sentence": "But the staff was so horrible to us .",
  "triplets": [
    {
      "aspect":          "staff",
      "aspect_indices":  [2],
      "opinion":         "horrible",
      "opinion_indices": [5],
      "sentiment":       "NEG"
    }
  ],
  "source": "rest14"
}
```

`*_indices` are zero-indexed positions over the whitespace-tokenised
sentence.  `sentiment` ∈ {POS, NEG, NEU}.

## Expected layout

After conversion the directory should contain:

```
data/
├── aste/
│   ├── rest14_train.jsonl  rest14_dev.jsonl  rest14_test.jsonl
│   ├── lap14_train.jsonl   lap14_dev.jsonl   lap14_test.jsonl
│   ├── rest15_train.jsonl  rest15_dev.jsonl  rest15_test.jsonl
│   └── rest16_train.jsonl  rest16_dev.jsonl  rest16_test.jsonl
├── parsed/                 # Stanza dependency parses, produced by
│   └── *_parsed.jsonl      #   `python scripts/parse_dependency.py`
└── distill_train.jsonl     # Distillation set with teacher rationales,
                            # produced by
                            #   `python scripts/generate_cot.py --model glm5`
                            # then
                            #   `python scripts/filter_cot.py`
```
