import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np
import re
import spacy
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load('en_core_web_sm')
stop_words = set(nltk.corpus.stopwords.words('english'))

def extract_text_from_pdf(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            cleaned_text = clean_text(text)
            pages_text.append((i + 1, cleaned_text))
    return pages_text

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', '. ', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1. \2', text)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'(\w)([A-Z])', r'\1. \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    print(text)
    return text

def tokenize_sentences(pages_text):
    sentences = []
    sentence_positions = []
    for page_num, text in pages_text:
        page_sentences = nltk.sent_tokenize(text)
        page_sentences = [sentence.strip() for sentence in page_sentences if sentence.strip()]
        sentences.extend(page_sentences)
        sentence_positions.extend([(page_num, i) for i in range(len(page_sentences))])
    return sentences, sentence_positions

def filter_important_words(sentence):
    doc = nlp(sentence)
    important_words = [token.text for token in doc if token.text.lower() not in stop_words and (token.ent_type_ or token.pos_ in {'NOUN', 'VERB', 'PROPN', 'NUM'})]
    return ' '.join(important_words)

def compute_lexical_similarity(reference_sentences, candidate_sentence):
    filtered_candidate = filter_important_words(candidate_sentence)
    filtered_references = [filter_important_words(sentence) for sentence in reference_sentences]
    vectorizer = TfidfVectorizer().fit_transform(filtered_references + [filtered_candidate])
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)
    return cosine_similarities[-1][:-1]

def compute_semantic_similarity_bertscore(reference_sentence, candidate_sentence):
    filtered_candidate = filter_important_words(candidate_sentence)
    filtered_reference = filter_important_words(reference_sentence)
    P, R, F1 = score([filtered_reference], [filtered_candidate], lang="en", rescale_with_baseline=True)
    return F1.numpy()[0]

def find_most_similar_sentence(reference_sentences, candidate_sentence, sentence_positions):
    lexical_similarities = compute_lexical_similarity(reference_sentences, candidate_sentence)
    most_similar_lexical_index = np.argmax(lexical_similarities)
    most_similar_lexical = reference_sentences[most_similar_lexical_index]
    lexical_score = lexical_similarities[most_similar_lexical_index]
    
    semantic_score_bertscore = compute_semantic_similarity_bertscore(most_similar_lexical, candidate_sentence)
    
    combined_score = (lexical_score + semantic_score_bertscore) / 2

    return most_similar_lexical, lexical_score, semantic_score_bertscore, combined_score, sentence_positions[most_similar_lexical_index]

def process_candidate_sentences(pdf_path, candidates_path, results_path):
    pages_text = extract_text_from_pdf(pdf_path)
    reference_sentences, sentence_positions = tokenize_sentences(pages_text)
    
    candidates_df = pd.read_excel(candidates_path)
    results = []
    
    for _, row in candidates_df.iterrows():
        candidate_sentence = row['Candidate']
        most_similar_lexical, lexical_score, semantic_score_bertscore, combined_score, position = find_most_similar_sentence(reference_sentences, candidate_sentence, sentence_positions)
        
        page_num, sentence_index = position
        results.append({
            'CandidateSentence': candidate_sentence,
            'MostSimilarSentence': most_similar_lexical,
            'LexicalSimilarityScore': lexical_score,
            'SemanticSimilarityScore_BERTScore': semantic_score_bertscore,
            'CombinedScore': combined_score,
            'PageNumber': page_num,
            'SentenceIndex': sentence_index
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(results_path, index=False)
    print(f"Results saved to {results_path}")

# Example usage
pdf_path = 'resources/2023-All.pdf'
candidates_path = 'resources/Candidates_set3.xlsx'
results_path = 'resources/results_set3_test1.xlsx'

if __name__ == "__main__":
    process_candidate_sentences(pdf_path, candidates_path, results_path)
