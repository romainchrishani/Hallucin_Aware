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
text1 = "How many departments are there in the faculty?"
text2 = "here are three departments in the faculty.14  Message from the Head, Department of Computational Mathematics  Welcome to the Faculty of Computational Mathematics, University of Moratuwa!  On behalf of the Department of Computational Mathematics, I would like to warmly welcome you to the faculty. It is a great pleasure to see hundreds of determined and dedicated young adults entrusting their future with the Faculty of Computational Mathematics. As  you begin your academic career in this prestigious institution, we congratulate you on your achievement, and your insight in choosing a program with high demand in this rapidly evolving discipline.  Department of Computational Mathematics remains one of the main academic departments providing the nation with professionally qualified mathematicians, scientists, and researchers in the fields of Computational Mathematics, Computer Science, and Information Technology. The curricula encompass a wide variety of subjects in Computational Mathematics, Computer Science, and Information Technology disciplines to provide both theoretical knowledge and practical exposure. Furthermore, the Department sets high emphasis on research studies and group work.  The Department maintains an unwavering reputation for its contribution in presenting academically sound, competent, and high -quality graduates to the workforce in the fields of Computational Mathematics, Computer Science, and Information Technology. There are quite a considerable number of graduates securing higher studies opportunities and scholarships in top -ranking international universities, immediately after graduation. We were also fortunate to produce several IT entrepreneurs whose startup company has grown into highly reputed award -winning companies with international recognition.  We, the Department of Computational Mathematics, encourage you to envision your future today, explore opportunities, embrace diversity, and be competent individuals with direction.  Wish you all a memorable and inspiring stay at the University of Moratuwa!  Mrs. Wijewardene  Head, Department of Computational Mathematics  Tel - office:  0112 -650894 ext.8200  web"


similarity = compute_cosine_similarity(text1, text2)
print(f"Cosine Similarity: {similarity}")


'''The cosine similarity score ranges from -1 to 1, where:

1 indicates that the texts are identical.
0 indicates no similarity.
-1 indicates complete dissimilarity.'''
