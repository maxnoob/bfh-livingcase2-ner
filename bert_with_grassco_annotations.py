from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rich import print
from rich.text import Text

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mschiesser/ner-bert-german")
model = AutoModelForTokenClassification.from_pretrained("mschiesser/ner-bert-german")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# previously used: dslim/bert-large-NER, HUMADEX/german_medical_ner, mschiesser/ner-bert-german

# Read text file
text_file = "Albers.txt"
with open("./Datasets/GraSSCo/corpus/%s"%text_file, "r", encoding="utf-8") as f:
    text = f.read()

# Model properties
max_length = tokenizer.model_max_length  # Typically 512 for BERT
print(f"Model's max input length: {max_length} tokens")
stride = max_length // 2  # Overlap for smooth transition
print(f"Used stride: {stride}")

# Process the text in chunks
ner_results = []
seen_spans = set()  # Track processed spans

print("Processing %s" % text_file)
for i in range(0, len(text), stride):
    chunk = text[i:i + max_length]
    chunk_results = nlp(chunk)
    # Display the processed token count on the same line
    print(f"Processed tokens: {i}", end='\r', flush=True)
    # Adjust start/end positions relative to the full text
    for result in chunk_results:
        start = result['start'] + i
        end = result['end'] + i
        
        # Avoid duplicate processing
        if (start, end, result['entity']) not in seen_spans:
            seen_spans.add((start, end, result['entity']))
            ner_results.append({
                'entity': result['entity'],
                'score': result['score'],
                'start': start,
                'end': end,
                'word': text[start:end]
            })
# print raw recognized entities
print("\n%s"%ner_results)


# take the original text and highlight the entities
highlighted_text = Text(text)

# see what entities the model can recognize and assign random colors to them
from random import shuffle
entities = set(label.split("-")[-1] for label in model.config.id2label.values() if label != "O")
available_colors = ["green", "blue", "red", "yellow", "magenta", "cyan"]
shuffle(available_colors)
# Create a color mapping
color_mapping = dict(zip(entities, available_colors[:len(entities)]))
# Print the color mapping
print("[bold underline]Assigned Colors for Labels:[/bold underline]")
for label, color in color_mapping.items():
    print(f"[{color}]{label}: {color}[/]")

# Highlight entities in the text using assigned colors
for entity in sorted(ner_results, key=lambda x: x['start']):
    entity_type = entity['entity'].split("-")[-1]  # Remove prefix
    color = color_mapping.get(entity_type, "cyan")  # Default to cyan
    
    # Add highlight to the text
    highlighted_text.stylize(color, entity['start'], entity['end'])

# Print the clean, highlighted text
print("\n[bold underline]Annotated Text:[/bold underline]")
print(highlighted_text)



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