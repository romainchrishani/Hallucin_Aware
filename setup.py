from setuptools import setup

REQUIREMENTS = [
    "transformers==4.40.0",
    "PyPDF2==3.0.1",
    "numpy",
    "bert_score",
    "spacy",
    "sentence_transformers",
    "faiss_cpu==1.8.0",
    "nltk",
    "torch>=1.12",
    "sentencepiece",
    "tqdm"
]

VERSION = {}
with open("hallucinaware/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="hallucinaware",
    version=VERSION["__version__"],
    py_modules=["hallucinaware"],
    description="HallucinAware: Detecting hallucination of responses from LLMs",
    author="Chrishani Perera",
    author_email="chrish.romi16@gmail.com",
    url="https://github.com/romainchrishani",
    license="MIT",
    include_package_data=True,
    install_requires=REQUIREMENTS,
    keywords="hallucinaware",
)
