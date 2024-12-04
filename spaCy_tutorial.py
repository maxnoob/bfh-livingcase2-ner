""" install spaCy
    Bevor Installation check, ob / welches environment verwendet wird
	python -m venv .env # create venv in project folder
	source .venv/bin/activate # activate venv
	
	pip install -U pip setuptools wheel
	pip install -U 'spacy[transformers,lookups]'
	
    install models:
    python3 -m spacy download en_core_web_trf # transformer version
    python3 -m spacy download en_core_web_sm # small version
    python3 -m spacy download de_dep_news_trf 
    
    check installed models and compatibility with spacy-version:
    python3 -m spacy validate
    
    info about installation:
    python -m spacy info
    """

import spacy
#import en_core_web_trf

# ---- Read in example text and split it by chapters ----
with open ("./alice_in_wonderland_text.txt","r") as f:
    text = f.read().replace("\n", " ") # replace line breaks, so that sentences stay coherent.
    # print(text) # print whole text file
    chapters = text.split("CHAPTER ")[1:]
    print("Number of chapters: %s" % len(chapters)) # print number of chapters (should be 12)
    # print (chapters[2]) # print a chapter based on index

chapter1 = chapters[0]
# spacy.cli.download("en_core_web_sm") # took about 10 minutes für 12 mb to load!...
# spacy.cli.download("en_core_web_lg") # didn't finish, even after over an hour
nlp = spacy.load("en_core_web_sm") # für deutsch-accuracy: de_dep_news_trf

doc = nlp(chapter1)
sentences = list(doc.sents)
print("First sentence: %s" % sentences[0]) # sentences[0] is the title

# ---- Working with Ents (predefined entities) -----
# With sm model a lot of false positive entities (takes most capitalized words as ent)
ents = list(doc.ents)
print("All entities of the text: %s" % ents)

# Ents are stored as tuples with metadata (label, label_, text)
# If we only want to see entities labeled as person:
people = []
for ent in ents:
    if ent.label_ == "PERSON":
        people.append(ent)
        
print("Only entities labeled 'PERSON':\n%s" % list(people))

# ---- Working with Tokens --------
tokens = []
for token in doc:
    token_tuple = ("%s, %s" % (token.text, token.pos_))
    tokens.append(token_tuple)
    #print("All tokens:\n%s" % tokens) # pos: part of speech (Verb, noun etc)
    
nouns = []
for token in doc:
    if token.pos_ == "NOUN":
        nouns.append(token)

print("All noun tokens:\n%s" % nouns)
# to get articles (a, its etc) use noun_chunks
print("Noun chunks:\n%s" % list(doc.noun_chunks))