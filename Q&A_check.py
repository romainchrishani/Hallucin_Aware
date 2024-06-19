from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(text1, text2):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the texts to TF-IDF matrices
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute the cosine similarity between the first and second text
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

# Example usage
text1 = "If a student miss an examination whom should he/she inform?"
text2 = "My name is Chrisani"

similarity = compute_cosine_similarity(text1, text2)
print(f"Cosine Similarity: {similarity}")


'''The cosine similarity score ranges from -1 to 1, where:

1 indicates that the texts are identical.
0 indicates no similarity.
-1 indicates complete dissimilarity.'''
