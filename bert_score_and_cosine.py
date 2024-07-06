#bertscore + cosine
import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from transformers import BertModel, BertTokenizer
from bert_score import score
import numpy as np

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    print(sentences)
    return sentences

def compute_lexical_similarity(reference_sentences, candidate_sentence):
    vectorizer = TfidfVectorizer().fit_transform(reference_sentences + [candidate_sentence])
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)
    return cosine_similarities[-1][:-1]

def compute_semantic_similarity_bertscore(reference_sentences, candidate_sentence):
    P, R, F1 = score(reference_sentences, [candidate_sentence] * len(reference_sentences), lang="en", rescale_with_baseline=True)
    return F1.numpy()

'''def compute_semantic_similarity_cosine(reference_sentences, candidate_sentence):
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
    '''

# Find the most similar sentence
def find_most_similar_sentence(reference_sentences, candidate_sentence):
    lexical_similarities = compute_lexical_similarity(reference_sentences, candidate_sentence)
    semantic_similarities_bertscore = compute_semantic_similarity_bertscore(reference_sentences, candidate_sentence)
    #semantic_similarities_cosine = np.array(compute_semantic_similarity_cosine(reference_sentences, candidate_sentence))

    most_similar_lexical = reference_sentences[np.argmax(lexical_similarities)]
    most_similar_semantic_bertscore = reference_sentences[np.argmax(semantic_similarities_bertscore)]
    #most_similar_semantic_cosine = reference_sentences[np.argmax(semantic_similarities_cosine)]

    return most_similar_lexical, max(lexical_similarities), most_similar_semantic_bertscore, max(semantic_similarities_bertscore)
    #, most_similar_semantic_cosine, max(semantic_similarities_cosine)


def main(pdf_path, candidate_sentence):
    text = extract_text_from_pdf(pdf_path)
    reference_sentences = tokenize_sentences(text)
    
    most_similar_lexical, lexical_score, most_similar_semantic_bertscore, semantic_score_bertscore= find_most_similar_sentence(reference_sentences, candidate_sentence)
    #, most_similar_semantic_cosine, semantic_score_cosine = find_most_similar_sentence(reference_sentences, candidate_sentence)
    
    print(f"Most similar sentence (Lexical Similarity): {most_similar_lexical}")
    print(f"Lexical Similarity Score: {lexical_score}")
    print(f"Most similar sentence (Semantic Similarity - BERTScore): {most_similar_semantic_bertscore}")
    print(f"Semantic Similarity Score (BERTScore): {semantic_score_bertscore}")
    #print(f"Most similar sentence (Semantic Similarity - Cosine): {most_similar_semantic_cosine}")
    #print(f"Semantic Similarity Score (Cosine): {semantic_score_cosine}")

# Example usage
pdf_path = 'resources/2023-All.pdf'
candidate_sentence = " The faculty of Information Technology was established in June 2001."
if __name__ == "__main__":
    main(pdf_path, candidate_sentence)

