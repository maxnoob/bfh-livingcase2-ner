from rich import print
from rich.text import Text
from query_model_info import get_full_model_info

# previously used:
# blaze999/Medical-NER (base: microsoft/deberta-v3-base) -> Model's max input length: 1000000000000000019884624838656 tokens?!
# NOTE: BERT models (e.g. HUMADEX/german_medical_ner) with 512 input size not working properly

model_name = "blaze999/Medical-NER"
tokenizer_name = model_name

loaded_model = get_full_model_info(model_name)
tokenizer = loaded_model["tokenizer"]
# get nlp pipeline
nlp = loaded_model["nlp_pipeline"]

# Calculate stride
max_length = loaded_model.get("tokenizers_max_token_length", 512) # use 512 as default if no max_length is found 
stride = max_length // 2
print(f"\nUsed stride: {stride}")

# Read text file
text_file = "Clausthal.txt"
with open(f"./data/GraSCCo/corpus/{text_file}", "r", encoding="utf-8") as f:
    text = f.read()

# Process the text in chunks
ner_results = []
seen_spans = set()  # Track processed spans

print(f"\nProcessing {text_file}")

def merge_entities(results):
    """Merge consecutive B- and I- entities into single entities."""
    merged = []
    current_entity = None

    for result in results:
        start, end = result['start'], result['end']
        label = result['entity'].split("-")[-1]  # Normalize label by removing B-/I-

        if current_entity is None or label != current_entity['entity'] or start != current_entity['end']:
            # Start a new entity
            if current_entity:
                merged.append(current_entity)
            current_entity = {
                'entity': label,
                'start': start,
                'end': end,
                'word': text[start:end],
                'score': result['score']
            }
        else:  # Merge with the current entity
            current_entity['end'] = end
            current_entity['word'] += text[start:end]
            current_entity['score'] = max(current_entity['score'], result['score'])

    if current_entity:
        merged.append(current_entity)
    return merged

for i in range(0, len(text), stride):
    chunk = text[i:i + max_length]
    chunk_results = nlp(chunk)
    print(f"Processed tokens: {i}", end='\r', flush=True)

    # Adjust start/end positions relative to the full text
    for result in chunk_results:
        start = result['start'] + i
        end = result['end'] + i

        merged_results = merge_entities(chunk_results)

    # Avoid duplicate spans
    for entity in merged_results:
        start, end, entity_type = entity['start'], entity['end'], entity['entity']
        if (start, end, entity_type) not in seen_spans:
            seen_spans.add((start, end, entity_type))
            ner_results.append(entity)

# Print raw recognized entities
print("\n[bold underline]Recognized Entities:[/bold underline]")
for result in ner_results:
    print(f"{result['word']} [{result['entity']}], Score: {result['score']:.2f}")

# Annotate the original text
annotated_text = Text()

last_end = 0
for entity in sorted(ner_results, key=lambda x: x['start']):
    start, end, entity_type = entity['start'], entity['end'], entity['entity']

    # Add text before the entity
    annotated_text.append(text[last_end:start])
    # Add the entity with its label in red
    annotated_text.append(f"{text[start:end]} ", style="bold yellow")
    annotated_text.append(f"[{entity_type}]", style="bold red")

    # Update the last processed position
    last_end = end

# Append remaining text
annotated_text.append(text[last_end:])

# Print annotated text
print("\n[bold underline]Annotated Text:[/bold underline]")
print(annotated_text)
