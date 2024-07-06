import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np
import re
import spacy

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
    combined_lexical_and_bertscore=(mean_lexical_score+semantic_score_bertscore)/2

    return most_similar_sentences, mean_lexical_score, semantic_score_bertscore, most_similar_positions, combined_lexical_and_bertscore

def main(pdf_path, candidate_text):
    pages_text = extract_text_from_pdf(pdf_path)
    reference_sentences, sentence_positions = tokenize_sentences(pages_text)
    candidate_sentences = nltk.sent_tokenize(candidate_text)
    
    most_similar_sentences, mean_lexical_score, semantic_score_bertscore, positions, combined_lexical_and_bertscore = find_most_similar_sentence(reference_sentences, candidate_sentences, sentence_positions)
    
    print(f"Most similar sentences (Lexical Similarity): {most_similar_sentences}")
    print(f"Mean Lexical Similarity Score: {mean_lexical_score}")
    print(f"Semantic Similarity Score (BERTScore): {semantic_score_bertscore}")
    for i, (page_num, sentence_index) in enumerate(positions):
        print(f"Position in PDF for candidate sentence {i+1} - Page: {page_num}, Sentence Index: {sentence_index}")
    print(f"Combined Score (Lexical + BERTscore): {combined_lexical_and_bertscore}")


# Example usage
pdf_path = 'resources/2023.pdf'
candidate_text = "here are three departments in the faculty.14  Message from the Head, Department of Computational Mathematics  Welcome to the Faculty of Computational Mathematics, University of Moratuwa!  On behalf of the Department of Computational Mathematics, I would like to warmly welcome you to the faculty. It is a great pleasure to see hundreds of determined and dedicated young adults entrusting their future with the Faculty of Computational Mathematics. As  you begin your academic career in this prestigious institution, we congratulate you on your achievement, and your insight in choosing a program with high demand in this rapidly evolving discipline.  Department of Computational Mathematics remains one of the main academic departments providing the nation with professionally qualified mathematicians, scientists, and researchers in the fields of Computational Mathematics, Computer Science, and Information Technology. The curricula encompass a wide variety of subjects in Computational Mathematics, Computer Science, and Information Technology disciplines to provide both theoretical knowledge and practical exposure. Furthermore, the Department sets high emphasis on research studies and group work.  The Department maintains an unwavering reputation for its contribution in presenting academically sound, competent, and high -quality graduates to the workforce in the fields of Computational Mathematics, Computer Science, and Information Technology. There are quite a considerable number of graduates securing higher studies opportunities and scholarships in top -ranking international universities, immediately after graduation. We were also fortunate to produce several IT entrepreneurs whose startup company has grown into highly reputed award -winning companies with international recognition.  We, the Department of Computational Mathematics, encourage you to envision your future today, explore opportunities, embrace diversity, and be competent individuals with direction.  Wish you all a memorable and inspiring stay at the University of Moratuwa!  Mrs. Wijewardene  Head, Department of Computational Mathematics  Tel - office:  0112 -650894 ext.8200  web"
if __name__ == "__main__":
    main(pdf_path, candidate_text)
