import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np
import re
import pandas as pd

nltk.download('punkt')

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

def compute_lexical_similarity(reference_sentences, candidate_sentence):
    vectorizer = TfidfVectorizer().fit_transform(reference_sentences + [candidate_sentence])
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)
    return cosine_similarities[-1][:-1]

def compute_semantic_similarity_bertscore(reference_sentence, candidate_sentence):
    P, R, F1 = score([reference_sentence], [candidate_sentence], lang="en", rescale_with_baseline=True)
    return F1.numpy()[0]

def find_most_similar_sentence(reference_sentences, candidate_sentence, sentence_positions):
    lexical_similarities = compute_lexical_similarity(reference_sentences, candidate_sentence)
    most_similar_lexical_index = np.argmax(lexical_similarities)
    most_similar_lexical = reference_sentences[most_similar_lexical_index]
    lexical_score = lexical_similarities[most_similar_lexical_index]
    
    semantic_score_bertscore = compute_semantic_similarity_bertscore(most_similar_lexical, candidate_sentence)
    
    combined_score = (lexical_score + semantic_score_bertscore) / 2

    return most_similar_lexical, lexical_score, semantic_score_bertscore, combined_score, sentence_positions[most_similar_lexical_index]

def process_candidates(pdf_path, candidates_excel_path, output_excel_path):
    # Extract text and tokenize sentences
    pages_text = extract_text_from_pdf(pdf_path)
    reference_sentences, sentence_positions = tokenize_sentences(pages_text)

    # Read candidate sentences from Excel
    df = pd.read_excel(candidates_excel_path)

    results = []

    for candidate_sentence in df['Candidate']:
        most_similar_lexical, lexical_score, semantic_score_bertscore, combined_score, position = find_most_similar_sentence(reference_sentences, candidate_sentence, sentence_positions)
        page_num, sentence_index = position
        results.append({
            'Candidate Sentence': candidate_sentence,
            'Most Similar Sentence': most_similar_lexical,
            'Lexical Similarity Score': lexical_score,
            'Semantic Similarity Score (BERTScore)': semantic_score_bertscore,
            'Combined Weighted Score': combined_score,
            'Page Number': page_num,
            'Sentence Index': sentence_index
        })

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_excel_path, index=False)

# Example usage
pdf_path = 'resources/2023-All.pdf'
candidates_excel_path = 'resources/candidates_All.xlsx'
output_excel_path = 'resources/results.xlsx'

if __name__ == "__main__":
    process_candidates(pdf_path, candidates_excel_path, output_excel_path)

