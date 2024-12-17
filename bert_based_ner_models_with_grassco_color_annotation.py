from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import sys
from rich import print
from rich.text import Text
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Load model and tokenizer
model_name = "mschiesser/ner-bert-german"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# previously used:
# dslim/bert-large-NER,
# HUMADEX/german_medical_ner,
# mschiesser/ner-bert-german: not too bad for person names, sees some diagnoses as orgniasations and medications as locations
# domischwimmbeck/bert-base-german-cased-fine-tuned-ner (434 MB) -> error while loading the model


# Read text file
text_file = "Clausthal.txt"
with open(f"./Datasets/GraSSCo/corpus/{text_file}", "r", encoding="utf-8") as f:
    text = f.read()
    
# Grassco corpus was annotated with entities:
# NAME PATIENT, NAME DOCTOR, NAME RELATIVE, NAME USERNAME 1, NAME TITLE, NAME EXTERN,
# DATE,
# LOCATION STREET, LOCATION ZIP, LOCATION CITY, LOCATION COUNTRY, LOCATION HOSPITAL, LOCATION ORGANIZATION,
# CONTACT PHONE 1, CONTACT FAX, CONTACT EMAIL,
# PROFESSION

# Extract model properties
model_properties = {
    "Model name": getattr(model, 'name_or_path', 'Unknown'),
    "Base model": model.config.architectures if hasattr(model.config, "architectures") else "Unknown",
    "Model type": model.config.model_type if hasattr(model.config, "model_type") else "Unknown",
    "Number of labels": model.config.num_labels if hasattr(model.config, "num_labels") else "Unknown", # label number is higher that entity number ('B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O')
    "Tokenizer name": tokenizer.name_or_path,
    "Tokenizer's max input length (# of tokens)": tokenizer.model_max_length, # Typically 512 for BERT
}

# Print extracted properties
print("\n[bold underline]Model properties[/bold underline]")
for prop, value in model_properties.items():
    print(f"{prop}: {value}")

# Calculate stride
max_length = model_properties.get("Tokenizer's max input length (# of tokens)", 512) # use 512 as default if no max_length is found 
stride = max_length // 2
print(f"Used stride: {stride}")

# Extract supported entities from the model
supported_entities = set()
for label in model.config.id2label.values():
    if label != "O":
        supported_entities.add(label.split("-")[-1])

# Print supported entities
print("\n[bold underline]Supported Entities by the Model:[/bold underline]")
for entity in sorted(supported_entities):
    print(f"• {entity}")

# Process the text in chunks
ner_results = []
seen_spans = set()  # Track processed spans

print("\nProcessing %s" % text_file)
for i in range(0, len(text), stride):
    chunk = text[i:i + max_length]
    chunk_results = nlp(chunk)
    # Display the processed token count on the same line
    print(f"Processed tokens: {i}", end='\r', flush=True)
    # Adjust start/end positions relative to the full text
    for result in chunk_results:
        span = (result['start'] + i, result['end'] + i, result['entity'])
        
        # Avoid duplicate processing
        if span not in seen_spans:
            seen_spans.add(span)
            ner_results.append({
                'entity': result['entity'],
                'score': result['score'],
                'start': span[0],
                'end': span[1],
                'word': text[span[0]:span[1]]
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

# ----- Draw confusion matrix ------
print("Processing whole corpus")

def normalize_predicted_entities(predicted_entities):
    """
    Normalizes predicted entities so that they can be compared into the form:
    {
    "start": int,
    "end": int,
    "entity": str
    }
    Merges consecutive B- and I- entities.
    """
    normalized = []
    current_entity = None

    for entity in predicted_entities:
        entity_type = entity["entity"].split('-')[-1]
        tag = entity["entity"].split('-')[0]

        if tag == "B" or (current_entity and current_entity["entity"] != entity_type):
            if current_entity:
                normalized.append(current_entity)
            current_entity = {
                "start": entity["start"],
                "end": entity["end"],
                "entity": entity_type
            }
        elif tag == "I" and current_entity:
            current_entity["end"] = entity["end"]

    if current_entity:
        normalized.append(current_entity)

    return normalized

def map_annotations(annotations, entity_mapping):
    """
    Maps annotations to a standardized entity type using the provided mapping.
    If the true entity is not in the predicted entity, then it's set to "OTHER".
    """
    return [
        {"start": ann["start"], "end": ann["end"], "entity": entity_mapping.get(ann["entity"], "OTHER")}
        for ann in annotations
    ]
    
# Align spans by comparing text overlap
def align_spans(true_spans, pred_spans):
    """
    Checks for overlapping entities of true and predicted annotations.
    This is done to compensate for not exact matches, since the annotated predictions are very bad (e.g. consecutive B-segments instead of B-, I-, I-).
    """
    true_labels = []
    predicted_labels = []
    
    for true_span in true_spans:
        true_start, true_end, true_label = true_span['start'], true_span['end'], true_span['entity']
        matched = False
        
        for pred_span in pred_spans:
            pred_start, pred_end, pred_label = pred_span['start'], pred_span['end'], pred_span['entity']
            
            # Check for overlap
            if not (pred_end <= true_start or pred_start >= true_end):
                true_labels.append(true_label)
                predicted_labels.append(pred_label)
                matched = True
        
        # Handle unmatched true spans
        if not matched:
            true_labels.append(true_label)
            predicted_labels.append("OUTSIDE")
    
    # Handle unmatched predicted spans
    for pred_span in pred_spans:
        pred_start, pred_end, pred_label = pred_span['start'], pred_span['end'], pred_span['entity']
        if not any(
            not (pred_end <= ts['start'] or pred_start >= ts['end'])
            for ts in true_spans
        ):
            true_labels.append("OUTSIDE")
            predicted_labels.append(pred_label)
    
    return true_labels, predicted_labels



def f3_score(y_true, y_pred, labels):
    """
    Calculates the F3-score (beta=3) to weigh the false negatives (entities that were missed for de-identification) more.
    https://towardsdatascience.com/is-f1-the-appropriate-criterion-to-use-what-about-f2-f3-f-beta-4bd8ef17e285
    """
    from sklearn.metrics import fbeta_score
    f_score = fbeta_score(y_true, y_pred, labels=labels, beta=3,  average='weighted', zero_division=0)
    return f_score


import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Map the entities from model and text corpus (unmapped true entities get mapped to OTHER in the map_annotations-function)
entity_mapping = {
    "NAME_PATIENT": "PER",
    "NAME_DOCTOR": "PER",
    "NAME_RELATIVE": "PER",
    "NAME_USERNAME_1": "PER",
    "NAME_TITLE": "PER",
    "NAME_EXTERN": "PER",
    "LOCATION_STREET": "LOC",
    "LOCATION_ZIP": "LOC",
    "LOCATION_CITY": "LOC",
    "LOCATION_COUNTRY": "LOC",
    "LOCATION_HOSPITAL": "ORG",
    "LOCATION_ORGANIZATION": "ORG"
}

# Batch file processing
input_dir = "./Datasets/GraSSCo/annotation/grascco_phi_annotation_json/"
json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# Initialize evaluation containers
true_entities, predicted_entities = [], []
total_true_entities_count = 0
total_predicted_entities_count = 0

# Process each JSON file
for json_file in json_files:
    try:
        with open(os.path.join(input_dir, json_file), "r", encoding="utf-8") as f:
            json_data = json.load(f)
            text = None
            annotations = []
            for feature in json_data['%FEATURE_STRUCTURES']:
                if text is None and feature.get('%TYPE') == "uima.cas.Sofa":
                    text = feature.get('sofaString')
                if feature.get('%TYPE') == "webanno.custom.PHI":
                    start = feature['begin']
                    end = feature['end']
                    label = feature.get('kind', 'PHI')
                    annotations.append({"start": start, "end": end, "entity": label})

            if not text or not text.strip():
                print(f"[red]Error: sofaString is empty in file {json_file}[/red]")
                continue

    except Exception as e:
        print(f"[red]Error reading {json_file}: {e}[/red]")
        continue

    # Recognize entities in the text
    ner_results = []
    seen_spans = set()
    max_length = tokenizer.model_max_length
    stride = max_length // 2
    
    # Process text in chunks to predict entities
    for i in range(0, len(text), stride):
        chunk = text[i:i + max_length]
        chunk_results = nlp(chunk)
        for result in chunk_results:
            span = (result['start'] + i, result['end'] + i, result['entity'])
            if span not in seen_spans:
                seen_spans.add(span)
                ner_results.append(result)
    # FOR DEBUGGING:
    # print(f"\nseen spans: {seen_spans}")
    # print(f"\nner_results: {ner_results}")
    print(f"File: {json_file}, predicted entities: {len(ner_results)}")

    # Extract predicted entities with spans
    predicted_spans = normalize_predicted_entities(ner_results)
    # FOR DEBUGGING:
    # print(f"predicted spans: {predicted_spans}")
    
    # Extract true entities from annotations
    print(f"File: {json_file}, true entities: {len(annotations)}")
    # FOR DEBUGGING:
    # print(f"annotations {annotations}")

    # Compare entities using the compare_entities function
    mapped_annotations = map_annotations(annotations, entity_mapping)
    
    # Update global counts before alignment
    total_true_entities_count += len(mapped_annotations)
    total_predicted_entities_count += len(predicted_spans)
    
    # Align true and predicted labels
    # if no match exists, "OUTSIDE" gets added to the other group, so that true and predicted entities can be compared later
    # this makes entity count seem higher, which is why we need an additional counter per group for the total count
    true_labels, predicted_labels = align_spans(mapped_annotations, predicted_spans)
    # FOR DEBUGGING:
    # print(f"mapped annotations: {mapped_annotations}")

    # Accumulate results across files
    true_entities.extend(true_labels)
    predicted_entities.extend(predicted_labels)

print(f"Total predicted entities: {total_predicted_entities_count}, Total true entities: {total_true_entities_count}")

# Define the labels to be considered ('OTHER' doesn't exist for model entities, but is neccessary for matrix)
labels = ["PER", "LOC", "ORG", "OTHER", "OUTSIDE"]

# Compute confusion matrix
conf_matrix = confusion_matrix(true_entities, predicted_entities, labels=labels)

# Create a DataFrame for visualization
conf_matrix_df = pd.DataFrame(
    conf_matrix, 
    index=[f"{label} (True)" for label in labels], 
    columns=[f"{label} (Pred)" for label in labels]
)

# Display the confusion matrix
print("\n[bold underline]Confusion matrix:[/bold underline]")
print(conf_matrix_df)


# Plot confusion matrix (for illustration purposes)
""" plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.show() """
   
# Print classification report
print("\n[bold underline]Classification Report:[/bold underline]")
print(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0)) # 0 instead of undefined for div by zero
# weighted average takes the number of true instances of each class into account.
# if the classes are imbalanced in the data, the weightet average should be considered

# Calculate and print F1 and F3 scores
overall_f1_score = f1_score(true_labels, predicted_labels, labels=labels,  average='weighted', zero_division=0)
print(f"[bold]Overall F1-score (weighted): {overall_f1_score:.4f}[/bold]")
overall_f3_score = f3_score(true_labels, predicted_labels, labels=labels)
print(f"[bold]Overall F3-score (weighted): {overall_f3_score:.4f}[/bold]")

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