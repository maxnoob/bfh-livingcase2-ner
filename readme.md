# Named Entity Recognition for De-Identification in Medical Texts

This repository is part of my project for the **Living Case 2 module** of the Bachelor’s program in Medical Informatics at the **Bern University of Applied Sciences (BFH)**. The project was developed and implemented by the repository owner.

## About this project
The goal of this project is to explore and implement Named Entity Recognition (NER) techniques for the de-identification of sensitive information** in medical texts. This involves the identification and removal of personal health information (PHI) such as names, dates, and locations.

### Limitations
- The code is provided "as-is" without any guarantees of performance or suitability for production use.
- It is meant for educational and exploratory purposes.



## Virtual environment
Create venv in project folder using the python version 3.9.13

```python3.9 -m venv .venv ```

Activate the environment venv

```source .venv/bin/activate```

(on Windows)

```.venv\Scripts\activate```

Install the requirements for your environment

```pip install -r requirements.txt```

## GraSCCo corpus
Since the annotated corpus is only about 13 MB in size, it is already to be found within this project.
If you want to make sure you have the most up to date version follow the steps below:

### To retrieve GraSCCo data (https://zenodo.org/records/6539131) run:

```pip3 install zenodo_get```

This will install zenodo_get from https://github.com/dvolgyes/zenodo_get. 

Then run

```mkdir GraSCCo```

```cd GraSCCo```

```zenodo_get 6539131```

## spaCy
To work with spaCy you can run the following commands, as per the documentation on https://spacy.io/usage.

```pip install -U pip setuptools wheel```

```pip install -U 'spacy[transformers,lookups]'```

```python3 -m spacy download en_core_web_trf```

```python3 -m spacy download de_dep_news_trf```

## License

This repository is licensed under the [MIT License](LICENSE), allowing anyone to freely use, modify, and distribute the code, provided proper attribution is given.

## Disclaimer

This project is intended for educational purposes and serves as a proof of concept. The results and methods should be thoroughly evaluated before being applied in any production or clinical context.

---

Feel free to contribute or raise any issues via the GitHub issue tracker.