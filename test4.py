import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Define functions for computing similarities
def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def compute_bert_similarity(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()

def compute_combined_similarity(text1, text2, weight_tfidf=0.5, weight_bert=0.5):
    tfidf_similarity = compute_tfidf_similarity(text1, text2)
    bert_similarity = compute_bert_similarity(text1, text2)
    combined_similarity = weight_tfidf * tfidf_similarity + weight_bert * bert_similarity
    return combined_similarity

# Load questions and answers from Excel file
input_file = 'questions_answers.xlsx'
output_file = 'similarity_results.xlsx'

df = pd.read_excel(input_file)

# Ensure the input file has columns 'Question' and 'Answer'
if 'Question' not in df.columns or 'Answer' not in df.columns:
    raise ValueError("Input Excel file must contain 'Question' and 'Answer' columns.")

results = []
threshold = 0.5  # Adjust as needed

# Iterate through the questions and answers
for index, row in df.iterrows():
    question = row['Question']
    generated_answer = row['Answer']
    
    similarity_score = compute_combined_similarity(question, generated_answer)
    
    relevance = "Relevant" if similarity_score >= threshold else "Not Relevant"
    results.append({
        'Question': question,
        'Generated Answer': generated_answer,
        'Similarity Score': similarity_score,
        'Relevance': relevance
    })

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Write the results to an Excel file
results_df.to_excel(output_file, index=False)

print(f"Results have been written to {output_file}")
