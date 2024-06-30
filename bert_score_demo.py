#working one, one by one
from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils
import numpy as np
import bert_score

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()
pdf_text = utils.read_pdf('resources/2023-5.pdf')
#pdf_text="The Department offers the first-ever BSc Hons in AI degree commenced with the 2021/2022 intake. Our department is dedicated to providing comprehensive knowledge of Artificial Intelligence and related areas, Mathematics, and Statistics for all bachelors’ degree programmes offered by the faculty. The Department’s dedicated staff comprises eight doctorly qualified and well-renowned experts in AI, Mathematics and Statistics, including one Senior Professor, 10 Senior Lecturers, one Junior Lecturer, three instructors, and five non-academic staff members."

reference_text = pdf_text

# Example candidate sentence
candidate = "According to the message of the dean, the mission of the university is to be the “Most globally recognized knowledge enterprise in South Asia”, currently maintaining a higher rank in both academic and research disciplines among the state universities in Sri Lanka as well as other higher educational institutions."

# Load spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_trf")
sentences = [sent for sent in nlp(reference_text).sents] 
reference_sentences = [sent.text.strip() for sent in sentences if len(sent) > 1]

# Calculate BERTScore similarity
candidate_scores = hallicinaware_bertscore.calculate_similarity(
    sentences=reference_sentences, 
    candidates=[candidate]
)

# Find the sentence with the highest BERTScore
max_index = np.argmax(candidate_scores)
most_similar_sentence = reference_sentences[max_index]
highest_f1_score = candidate_scores[max_index]

print(f"Most similar sentence: {most_similar_sentence}")
print(f"Highest F1 Score: { highest_f1_score}")  
print(f"Highest BERTScore F1: {1.0 - highest_f1_score}")  # since scores were returned as 1 - BERTScore


