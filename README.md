# Named Entity Recognition (NER) System (Transformers)

## Files
- `ner.py` — single script: loads Hugging Face's default pretrained NER pipeline, extracts
  entities with type + confidence score, runs one sample sentence, then an interactive CLI
- `requirements.txt`

## Run
```bash
pip install transformers torch
python ner.py
```

## Sample input/output
```
Sample text: Elon Musk founded Tesla in California in 2003.
Elon Musk -> PER (confidence: 0.99)
Tesla -> ORG (confidence: 0.97)
California -> LOC (confidence: 0.99)
```

Then it drops into an interactive loop:
```
Enter text (or 'exit'): Barack Obama was born in Hawaii and worked with Google.
Barack Obama -> PER (confidence: 0.99)
Hawaii -> LOC (confidence: 0.98)
Google -> ORG (confidence: 0.99)
```

## How it works
- `pipeline("ner", grouped_entities=True)` loads a pretrained Transformer NER model and
  its tokenizer together
- `grouped_entities=True` merges multi-token entities ("Elon" + "Musk") into a single
  result instead of returning them as separate word-piece tokens
- Each result includes the entity text, its type (PER/ORG/LOC/etc.), and a confidence score
- `extract_entities()` prints every entity found in a piece of text in `word -> TYPE` format
- `run_cli()` lets you keep entering new sentences until you type "exit"

No bonus features included (no dbmdz/bert-large-cased-finetuned-conll03-english override,
colored highlighting, Streamlit dashboard, spaCy comparison, CSV export, or confidence
filtering) per request.
