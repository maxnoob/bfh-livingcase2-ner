from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rich import print
from rich.text import Text

# Load model and tokenizer
model_name = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# previously used:
# blaze999/Medical-NER (base: microsoft/deberta-v3-base) -> Model's max input length: 1000000000000000019884624838656 tokens?!

# Read text file
text_file = "Clausthal.txt"
with open(f"./Datasets/GraSSCo/corpus/{text_file}", "r", encoding="utf-8") as f:
    text = f.read()

# Model properties
max_length = tokenizer.model_max_length
stride = max_length // 2  # Stride is half the max_length for overlapping chunks
print(f"\nModel's max input length: {max_length} tokens")
print(f"Used stride: {stride}")

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

# Print supported entities
supported_entities = set(label.split("-")[-1] for label in model.config.id2label.values() if label != "O")
print("\n[bold underline]Supported Entities:[/bold underline]")
for entity in sorted(supported_entities):
    print(f"• {entity}")

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



""" 
Vizualization of recognized ents with iPython in table quite ok. But colored rich text is easier to read.
--------------------------------------------------------------------
import pandas as pd
from IPython.display import display

# Process and organize the data
entity_data = [
    {
        "Entity Type": entity['entity'],
        "Text": text[entity['start']:entity['end']],
        "Score": entity['score'],
        "Start Position": entity['start'],
        "End Position": entity['end']
    }
    for entity in ner_results
]

# Create a DataFrame
df = pd.DataFrame(entity_data)

# Display the table
display(df) """


""" 
Vizualization with termocolor (coloring the annotated text in the terminal) caused problems with ANSI signs, that were generated and made the output partially unreadable
--------------------------------------------------------
from termcolor import colored

# Highlight entities in the text with different colors
highlighted_text = text
# If the entity isn't found in the dictionary of entities, get() returns "cyan" as the default
for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
    color = {
    "LOC": "green",
    "PER": "blue",
    "ORG": "red",
    "MISC": "yellow"
    }.get(entity['entity'].split("-")[-1], "cyan") # split because of the prefixes
    entity_text = text[entity['start']:entity['end']]
    highlighted_text = (
        highlighted_text[:entity['start']] +
        colored(entity_text, color) +
        highlighted_text[entity['end']:]
    ) """


""" 
Visualization with displacy from spacy (showing the annotated text in html) not working properly. To few ents get converted into spacy ents.
-----------------------------------------------------------------
from spacy.tokens import Span
from spacy.lang.de import German
from spacy import displacy

# Create a spaCy document
nlp_spacy = German()
doc = nlp_spacy(text)

# Map character offsets to token indices
ents = []
for ent in ner_results:
    label = ent['entity'].split("-")[-1]  # Remove prefixes like B- or I-
    span = doc.char_span(ent['start'], ent['end'], label=label)
    if span:
        ents.append(span)

# Assign entities: Check for overlaps and assign safely
if ents:
    try:
        # Try assigning to doc.ents (only if no overlaps)
        doc.ents = ents
        print(f"Assigned {len(doc.ents)} entities to `doc.ents`.")
        displacy.serve(doc, style="ent", options={"compact": False, "fine_grained": True, "limit": 0}, port=5000)
    except ValueError:
        # Handle overlapping entities using spans
        print("Overlapping entities detected, using `doc.spans['entities']`.")
        doc.spans["entities"] = ents
        
        # Render using displacy.render() with spans_key
        html = displacy.render(
            doc, style="ent", spans_key="entities", options={"compact": False, "fine_grained": True, "limit": 0}
        )
        
        from IPython.core.display import display, HTML
        display(HTML(html))
else:
    print("No entities found.")
 """