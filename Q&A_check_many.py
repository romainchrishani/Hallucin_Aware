from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def compute_cosine_similarity(text1, text2):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the texts to TF-IDF matrices
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute the cosine similarity between the first and second text
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

# Load the questions and answers from the Excel file
df = pd.read_excel('resources\Test.xlsx')

# Assuming the column names are 'Question' and 'Answer'
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Prepare a list to store results
results = []

# Iterate over the questions and answers to compute cosine similarity
for question, answer in zip(questions, answers):
    similarity = compute_cosine_similarity(question, answer)
    results.append((question, answer, similarity))

# Create a DataFrame from results and save to a new Excel file
results_df = pd.DataFrame(results, columns=['Question', 'Answer', 'Cosine Similarity'])
results_df.to_excel('resources/results_cosine_similarity.xlsx', index=False)

print("Results have been saved to 'resources/results_cosine_similarity.xlsx'")
