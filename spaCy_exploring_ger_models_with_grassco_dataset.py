""" 
This Script is for exploring text data with spacy models to do NER.
In this case with the model de_core_news_lg, since the model de_dep_news_trf does not contain
a NER pipeline.

install spaCy
    before installation, check the used environment
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

import spacy

text_file = "Albers.txt"

# ---- Read in example text  ----
with open (f"./Datasets/GraSSCo/corpus/{text_file}","r") as f:
    text = f.read().replace("\n", " ") # replace line breaks, so that sentences stay coherent.
    # print(text) # print whole text file

# ----- also tried older python and spacy version --------
""" working for now with this stable setup
spaCy version    3.7.5                         
Platform         macOS-12.7.6-x86_64-i386-64bit
Python version   3.9.13                        
Pipelines        de_core_news_lg     """
# entity recognition and sentence splitting still not working in german. only recognizes nouns
# solution to model loading problems: upgrade spaCy with 'pip install -U spacy' 

# ---- models (german) ----

# transformer model (de_dep_news_trf)
# download the model:
# python -m spacy download de_dep_news_trf # 420 mb
# nlp = spacy.load("de_dep_news_trf")
# de_dep_news_trf does not have any NER labels and no NER pipeline, per documentation!
# Pipelines for the transformer model: transformer, tagger, morphologizer, parser, lemmatizer, attribute_ruler

# non-transformer models (de_core_news_sm/md/lg)
# download the model:
# python -m spacy download de_core_news_lg # 570 mb
# de_core_news_lg has the NER labels LOC, MISC, ORG, PER
# Non-transformer model has pipeplines: tok2vec, tagger, morphologizer, parser, lemmatizer, attribute_ruler, ner
nlp = spacy.load("de_core_news_lg")
# in non-transformer models you can disable everything except NER
# nlp = spacy.load("de_core_news_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe('sentencizer') # add the 'sentencizer' component to the pipeline

# since entity recognition did not work in german transformer model, tried different spacy-versions (didn't know german trf model doesn't have NER)
# needed to downgrade numpy to below 2.0, since runtime errors occured
# 'pip install "numpy<2.0" ' fixed this; but stuff not working properly
# 'pip install --upgrade torch' didn't change anything
# 'pip install --upgrade transformers' said spacy-transformers 1.3.5 requires transformers<4.37.0,>=3.4.0, but you have transformers 4.46.3 which is incompatible.
# 'pip install --upgrade 'transformers<4.37.0'

doc = nlp(text)
sentences = list(doc.sents)
print("\nFirst sentence: %s" % sentences[0]) # sentences[0] is the title

# the Ents and Tokens below work with the english models flawlessly
# with german models
# - Ents work only with de_core_news_lg (non-transformer) and after adding sentencizer to pipeline
# - Tokens work only with de_dep_news_trf

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

# -----> as seen by the output, the labeling is very poor.

for ent in ents:
    print("\n %s | %s" % (ent, ent.label_))

# ---- Working with Tokens (de_dep_news_trf can use tokens too) --------
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
print("\nNoun chunks:\n%s" % list(doc.noun_chunks))