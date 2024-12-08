""" 
This Script is for exploring text data with spacy models to do NER.
In this case with the model de_core_news_lg, since the model de_dep_news_trf does not contain
a NER pipeline.

install spaCy
    Bevor Installation check, ob / welches environment verwendet wird
	python -m venv .env # create venv in project folder
    (python3.10 -m venv .python3.10-spacy)
	source .venv/bin/activate # activate venv
    check python version:
    python --version
    (to deactive the current env, just type 'deactivate')
	
	pip install -U pip setuptools wheel
	pip install -U 'spacy[transformers,lookups]'

    install models:
    python -m spacy download de_dep_news_trf # throws warning due to problem with numpy-versions
    
    check installed models and compatibility with spacy-version:
    python3 -m spacy validate
    
    info about installation:
    python -m spacy info
    """

"""
    for german language tokenization:
    python -m spacy download de_dep_news_trf # around 400 mb
    """

import spacy
import json
import os
from spacy.tokens import DocBin


# ---- Read in example text  ----
with open ("./Datasets/GraSSCo/corpus/Albers.txt","r") as f:
    text = f.read().replace("\n", " ") # replace line breaks, so that sentences stay coherent.
    # print(text) # print whole text file

# ----- also tried older python and spacy version --------
""" ============================== Info about spaCy ==============================

spaCy version    3.7.5                         
Location         /Users/fabianburki/git/bfh_hs24/bfh-livingcase2-ner/.venv/lib/python3.9/site-packages/spacy
Platform         macOS-12.7.6-x86_64-i386-64bit
Python version   3.9.13                        
Pipelines        de_dep_news_trf (3.7.2)     """
# entity recognition and sentence splitting still not working in german. only recognizes nouns

# ---- models (german) ----

# transformer model (de_dep_news_trf)
# download the model:
# python -m spacy download de_dep_news_trf # 420 mb
# nlp = spacy.load("de_dep_news_trf")
# de_dep_news_trf does not have any NER labels and no NER pipeline, per documentation!
# Pipelines for the transformer model: transformer, tagger, morphologizer, parser, lemmatizer, attribute_ruler
""" 
# non-transformer models (de_core_news_sm/md/lg)
# download the model:
# python -m spacy download de_core_news_lg # 570 mb
# de_core_news_lg has the NER labels LOC, MISC, ORG, PER
# Non-transformer model has pipeplines: tok2vec, tagger, morphologizer, parser, lemmatizer, attribute_ruler, ner
nlp = spacy.load("de_core_news_lg")
# in non-transformer models you can disable everything except NER
# nlp = spacy.load("de_core_news_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe('sentencizer') # add the 'sentencizer' component to the pipeline

doc = nlp(text)
sentences = list(doc.sents)
print("\nFirst sentence: %s" % sentences[0]) # sentences[0] is the title

# ---- Working with Ents (predefined entities, working only with de_core_news_lg and after adding sentencizer to pipeline) -----
ents = list(doc.ents)
print("\nAll entities of the text: %s" % ents)

# Ents are stored as tuples with metadata (label, label_, text)
# If we only want to see entities labeled as person:
people = []
for ent in ents:
    if ent.label_ == "PER":
        people.append(ent)
        
print("\nEntities labeled 'PER':\n%s" % list(people))

locations = []
for ent in ents:
    if ent.label_ == "LOC":
        locations.append(ent)
        
print("\nEntities labeled 'LOC':\n%s" % list(locations))

organziations = []
for ent in ents:
    if ent.label_ == "ORG":
        organziations.append(ent)
        
print("\nEntities labeled 'ORG':\n%s" % list(organziations))

for ent in ents:
    print("\n %s | %s" % (ent, ent.label_))

# ---- Working with Tokens (working only with de_dep_news_trf) --------
tokens = []
for token in doc:
    token_tuple = ("%s, %s" % (token.text, token.pos_))
    tokens.append(token_tuple)
    #print("All tokens:\n%s" % tokens) # pos: part of speech (Verb, noun etc)
    
nouns = []
for token in doc:
    if token.pos_ == "NOUN":
        nouns.append(token)

print("\nAll noun tokens:\n%s" % nouns)
# to get articles (a, its etc) use noun_chunks
print("\nNoun chunks:\n%s" % list(doc.noun_chunks)) """


# --- read annotation files and convert them in spacy format ----
json_files = [
    "Albers",
    "Amanda_Alzheimer"
]
annotation_dir = "./Datasets/GraSSCo/annotation/grascco_phi_annotation_json"
training_dir = "./spacy_training_data"
os.makedirs(training_dir, exist_ok=True)

def parse_json_annotations_to_spacy_format(json_file):
    print("Parsing %s" % json_file)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the original text (sofaString) and the annotations
    text = None
    annotations = []
    for feature in data['%FEATURE_STRUCTURES']:
        if text == None:
            if feature.get('%TYPE') == "uima.cas.Sofa":
                text = feature.get('sofaString')
        if feature.get('%TYPE') == "webanno.custom.PHI":
            start = feature['begin']
            end = feature['end']
            label = feature.get('kind', 'PHI')  # Default label to PHI if not specified
            annotations.append((start, end, label))
    
    # Convert to spaCy format
    spacy_data = [(text, {"entities": annotations})]
    print(spacy_data) # preview the data
    return spacy_data


# --- save the annotations in the spacy format ----
# Function to create individual .spacy files
def create_individual_training_files(input_dir, output_dir):
    nlp = spacy.blank("de")  # Adjust language if necessary
    # Get all files in the folder
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(files)
    for file in files:
        json_file = os.path.join(input_dir, file)
        # Parse the JSON to spaCy format
        spacy_format_data = parse_json_annotations_to_spacy_format(json_file)
        
        # Create a DocBin and add the data
        db = DocBin()
        for text, annotations in spacy_format_data:
            doc = nlp.make_doc(text)
            spans = [
                doc.char_span(start, end, label) 
                for start, end, label in annotations["entities"]
            ]
            doc.ents = [span for span in spans if span is not None]
            db.add(doc)
        
        # Save the .spacy file with the name of the JSON file (without extension)
        output_path = f"{output_dir}/{file.replace('.json','.spacy')}"
        db.to_disk(output_path)

create_individual_training_files(annotation_dir, training_dir)