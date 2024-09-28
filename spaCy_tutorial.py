with open ("./alice_in_wonderland_text.txt","r") as f:
    text = f.read()
    # print(text) # print whole text file
    chapters = text.split("CHAPTER ")[1:]
    print (len(chapters)) # print number of chapters (should be 12)
    # print (chapters[2]) # print a chapter based on index
   
""" install spaCy
    Bevor Installation check, ob / welches environment verwendet wird
	python -m venv .env # create venv in project folder
	source .venv/bin/activate # activate venv
	
	pip install -U pip setuptools wheel
	pip install -U 'spacy[transformers,lookups]'
	python3 -m spacy download en_core_web_trf
    python3 -m spacy download de_dep_news_trf """
   # python3 -m spacy validate # to validate compatibility of models with spacy-version
    
import spacy
nlp = spacy.load("en_core_web_trf") # für deutsch-accuracy: de_dep_news_trf

chapter1 = chapters[0]
doc = nlp(chapter1)
sentences = list(doc.sents)
print(sentences[0])