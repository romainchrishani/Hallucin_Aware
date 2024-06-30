import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import re

class QAConfig:
    answering_bert: str = "bert-large-uncased-whole-word-masking-finetuned-squad"
    answering_bart: str = "facebook/bart-large"

def expand_list(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def duplicate_sentences(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
    
    cleaned_text = [clean_text(page) for page in text]
    cleaned_text = " ".join(cleaned_text)  
    return cleaned_text  

def clean_text(text):
    # Remove non-text elements
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove email addresses
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    # text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters

    text = re.sub(r'\.{2,}', '', text)

    # Normalize whitespace
    text = re.sub('\s+', ' ', text).strip()

    #text = text.replace('.', '')
    
    print("Text",text)
    return text


def encode_sentences(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings.numpy()

def encode_sentences_t5(documents):
    model = SentenceTransformer('sentence-t5-xl')
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings.numpy()


def create_embedding_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d) # IndexFlatL2 is suitable for Euclidean distance-based search
    index.add(embeddings)
    return index


def retrieve_similar_documents(query_embedding, index, documents, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    relevance_threshold = 1.2
    if np.min(distances) > relevance_threshold:
        return None
    return [documents[i] for i in indices[0]]


def setup_qa_system(
    answering_type: str = 'bert',
    ):
    
    if answering_type == 'bert':
        answering = QAConfig.answering_bert
    elif answering_type == "bart":
        answering = QAConfig.answering_bart
        
    tokenizer = AutoTokenizer.from_pretrained(answering)
    model = AutoModelForQuestionAnswering.from_pretrained(answering)
    qa_pipeline = pipeline(
        'question-answering',
        model=model, 
        tokenizer=tokenizer,
        max_length=512,
        top_k=5,
        threshold=None
        )
    return qa_pipeline


def answer_question_with_qa_system(question, context, qa_pipeline):
    inputs = {"question": question, "context": context}
    answers = qa_pipeline(inputs)
    print(answers)
    return answers

