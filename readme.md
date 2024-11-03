Create venv in project folder

```python3 -m venv .env ```

Activate the environment venv

```source .venv/bin/activate```


### To retrieve GraSSCo data (https://zenodo.org/records/6539131) run:

```pip3 install zenodo_get```

This will install zenodo_get from https://github.com/dvolgyes/zenodo_get. 

Then run

```mkdir GraSSCo```

```cd GraSSCo```

```zenodo_get 6539131```

### For spaCy:

```pip install -U pip setuptools wheel```

```pip install -U 'spacy[transformers,lookups]'```

```python3 -m spacy download en_core_web_trf```

```python3 -m spacy download de_dep_news_trf```