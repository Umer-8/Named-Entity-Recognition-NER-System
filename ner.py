from transformers import pipeline

ner_pipeline = pipeline("ner", grouped_entities=True)


def extract_entities(text):
    results = ner_pipeline(text)
    for entity in results:
        print(f"{entity['word']} -> {entity['entity_group']} (confidence: {entity['score']:.2f})")
    return results


def run_examples():
    sample_text = "Elon Musk founded Tesla in California in 2003."
    print("Sample text:", sample_text)
    extract_entities(sample_text)
    print("-" * 60)


def run_cli():
    while True:
        user_input = input("Enter text (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
        extract_entities(user_input)


if __name__ == "__main__":
    run_examples()
    run_cli()
