# Teacher Prompt Template

This is the exact prompt template that we feed to the teacher LLM (GLM-5
by default) when generating gold-guided chain-of-thought rationales for the
distillation training set.  The same template lives in
`scripts/generate_cot.py` (constants `SYSTEM_PROMPT` and
`build_user_prompt`); this file is duplicated here purely for
documentation / quick reference.

## System prompt

```
You are an expert in Aspect-Based Sentiment Analysis (ABSA).
Your task: Given a review sentence and its correct annotation
(aspect-opinion-sentiment triplets), generate a detailed step-by-step
reasoning process explaining WHY each triplet is correct.

Requirements:
1. Analyze each triplet one by one.
2. For each triplet, explain:
   - How the aspect term is identified (explicit mention or implicit inference).
   - How the opinion term expresses sentiment toward the aspect.
   - Why the sentiment polarity (POS/NEG/NEU) is assigned.
   - Note any linguistic cues: negation, contrast words (but/however),
     intensifiers, sarcasm.
3. If there are multiple triplets, explain their relationships
   (e.g., contrast, parallel).
4. Keep the reasoning concise but thorough (150-300 words).
5. End with: "Final Answer: [repeat the triplets exactly]"
```

## User prompt template

Filled per training example.  Triplets are formatted as a Python-list of
`(aspect, opinion, sentiment)` tuples.

```
Sentence: "{sentence}"

Correct Annotation (Gold Triplets): [("{aspect}", "{opinion}", {sentiment}), ...]

Please provide a step-by-step reasoning process explaining why these
triplets are correct.
```

## Notes

- We use `temperature=0.1` and `max_tokens=1024` when sampling from the
  teacher.
- Rationales that fail a downstream sanity filter (see
  `scripts/filter_cot.py` — checks that the final answer matches the gold
  triplets character-for-character) are dropped.  Empirically ~5-10% of
  rationales are filtered out.
