import pdfplumber
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from bert_score import score
import numpy as np

# Download the punkt tokenizer for NLTK
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Tokenize text into sentences
def tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

# Compute Lexical Similarity using TF-IDF
def compute_lexical_similarity(reference_sentences, candidate_sentence):
    vectorizer = TfidfVectorizer().fit_transform(reference_sentences + [candidate_sentence])
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)
    return cosine_similarities[-1][:-1]

# Compute Semantic Similarity using BERTScore
def compute_semantic_similarity_bertscore(reference_sentences, candidate_sentence):
    P, R, F1 = score(reference_sentences, [candidate_sentence] * len(reference_sentences), lang="en", rescale_with_baseline=True)
    return F1.numpy()

# Compute Semantic Similarity using Cosine Similarity with BERT embeddings
def compute_semantic_similarity_cosine(reference_sentences, candidate_sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def encode_sentence(sentence):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    candidate_embedding = encode_sentence(candidate_sentence)
    reference_embeddings = [encode_sentence(sentence) for sentence in reference_sentences]

    similarities = [cosine_similarity(candidate_embedding, ref_emb)[0][0] for ref_emb in reference_embeddings]
    return similarities

# Find the most similar sentence
def find_most_similar_sentence(reference_sentences, candidate_sentence):
    lexical_similarities = compute_lexical_similarity(reference_sentences, candidate_sentence)
    semantic_similarities_bertscore = compute_semantic_similarity_bertscore(reference_sentences, candidate_sentence)
    semantic_similarities_cosine = np.array(compute_semantic_similarity_cosine(reference_sentences, candidate_sentence))

    most_similar_lexical = reference_sentences[np.argmax(lexical_similarities)]
    most_similar_semantic_bertscore = reference_sentences[np.argmax(semantic_similarities_bertscore)]
    most_similar_semantic_cosine = reference_sentences[np.argmax(semantic_similarities_cosine)]

    return most_similar_lexical, max(lexical_similarities), most_similar_semantic_bertscore, max(semantic_similarities_bertscore), most_similar_semantic_cosine, max(semantic_similarities_cosine)

# Main function to process all candidate sentences
def process_candidate_sentences(pdf_path, excel_input_path, excel_output_path):
    text = extract_text_from_pdf(pdf_path)
    reference_sentences = tokenize_sentences(text)
    
    # Read candidate sentences from Excel file
    df = pd.read_excel(excel_input_path, sheet_name='Sheet1')
    candidate_sentences = df['Candidate'].tolist()
    
    results = []
    for candidate_sentence in candidate_sentences:
        most_similar_lexical, lexical_score, most_similar_semantic_bertscore, semantic_score_bertscore, most_similar_semantic_cosine, semantic_score_cosine = find_most_similar_sentence(reference_sentences, candidate_sentence)
        
        results.append({
            'Candidate Sentence': candidate_sentence,
            'Most Similar Lexical Sentence': most_similar_lexical,
            'Lexical Similarity Score': lexical_score,
            'Most Similar Semantic BERTScore Sentence': most_similar_semantic_bertscore,
            'Semantic BERTScore Similarity Score': semantic_score_bertscore,
            'Most Similar Semantic Cosine Sentence': most_similar_semantic_cosine,
            'Semantic Cosine Similarity Score': semantic_score_cosine
        })
    
    # Create DataFrame and write results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_output_path, index=False)

# Example usage
pdf_path = 'resources/2023-All.pdf'
excel_input_path = 'resources/Candidate8.xlsx'  # Excel file containing candidate sentences
excel_output_path = 'resources/similarity_results.xlsx'  # Output Excel file to save results

if __name__ == "__main__":
    process_candidate_sentences(pdf_path, excel_input_path, excel_output_path)
