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
            if text:
                cleaned_text = clean_text(text)
                table_sentences = extract_rows_from_table(cleaned_text)
                pages_text.append((i + 1, table_sentences))
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

def extract_rows_from_table(table_text):
    rows = table_text.split('. ')
    sentences = []
    for row in rows:
        columns = re.split(r'\s{2,}', row)
        if columns:
            sentence = ' '.join(columns).strip()
            if sentence:
                sentences.append(sentence)
    return sentences

def tokenize_sentences(pages_text):
    sentences = []
    sentence_positions = []
    for page_num, page_sentences in pages_text:
        for i, sentence in enumerate(page_sentences):
            sentences.append(sentence)
            sentence_positions.append((page_num, i))
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

def find_most_similar_sentence(reference_sentences, candidate_sentences, sentence_positions):
    all_lexical_scores = []
    most_similar_sentences = []
    most_similar_positions = []

    for candidate_sentence in candidate_sentences:
        lexical_similarities = compute_lexical_similarity(reference_sentences, candidate_sentence)
        most_similar_index = np.argmax(lexical_similarities)
        most_similar_sentence = reference_sentences[most_similar_index]
        lexical_score = lexical_similarities[most_similar_index]

        all_lexical_scores.append(lexical_score)
        most_similar_sentences.append(most_similar_sentence)
        most_similar_positions.append(sentence_positions[most_similar_index])

    mean_lexical_score = np.mean(all_lexical_scores)
    combined_similar_sentences = " ".join(most_similar_sentences)
    combined_candidate_sentences = " ".join(candidate_sentences)
    semantic_score_bertscore = compute_semantic_similarity_bertscore(combined_similar_sentences, combined_candidate_sentences)
    combined_lexical_and_bertscore = (mean_lexical_score + semantic_score_bertscore) / 2

    return most_similar_sentences, mean_lexical_score, semantic_score_bertscore, most_similar_positions, combined_lexical_and_bertscore

def process_excel(input_excel_path, output_excel_path, pdf_path):
    df = pd.read_excel(input_excel_path)
    results = []

    pages_text = extract_text_from_pdf(pdf_path)
    reference_sentences, sentence_positions = tokenize_sentences(pages_text)

    for index, row in df.iterrows():
        candidate_text = row['Candidate']
        candidate_sentences = nltk.sent_tokenize(candidate_text)
        
        most_similar_sentences, mean_lexical_score, semantic_score_bertscore, positions, combined_score = find_most_similar_sentence(reference_sentences, candidate_sentences, sentence_positions)
        
        for i, (page_num, sentence_index) in enumerate(positions):
            results.append([
                row['No'], 
                candidate_text, 
                candidate_sentences[i], 
                most_similar_sentences[i], 
                f"Page: {page_num}, Sentence Index: {sentence_index}",
                mean_lexical_score, 
                semantic_score_bertscore, 
                combined_score
            ])
    
    results_df = pd.DataFrame(results, columns=[
        'No', 'Candidate Text', 'Sentences of the candidate text', 'Most similar sentence', 'Position in PDF for candidate sentence',
        'Mean Lexical Similarity Score', 'Semantic Similarity Score (BERTScore)', 'Combined score'
    ])
    
    results_df.to_excel(output_excel_path, index=False)
    print(f"Results written to {output_excel_path}")

# Example usage
input_excel_path = 'resources/ChatHistory (1).xlsx'
output_excel_path = 'resources/Resultstest.xlsx'
pdf_path = 'resources/2023.pdf'

if __name__ == "__main__":
    process_excel(input_excel_path, output_excel_path, pdf_path)
