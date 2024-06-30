
from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version, utils
import spacy
import numpy as np
import bert_score
import re

# Print HallucinAware version
print("hallucin-aware version: ", version.__version__)

# Initialize HallucinAwareBERTScore
hallicinaware_bertscore = HallucinAwareBERTScore()

# Example reference text (from the PDF)
#reference_text = "The Department’s dedicated staff comprises eight doctorly qualified and well-renowned experts in AI, Mathematics and Statistics, including one Senior Professor, 10 Senior Lecturers, one Junior Lecturer, three instructors, and five non-academic staff members."

pdf_text = utils.read_pdf('resources/2023.pdf')
#pdf_text="The Department offers the first-ever BSc Hons in AI degree commenced with the 2021/2022 intake. Our department is dedicated to providing comprehensive knowledge of Artificial Intelligence and related areas, Mathematics, and Statistics for all bachelors’ degree programmes offered by the faculty. The Department’s dedicated staff comprises eight doctorly qualified and well-renowned experts in AI, Mathematics and Statistics, including one Senior Professor, 10 Senior Lecturers, one Junior Lecturer, three instructors, and five non-academic staff members."

reference_text = pdf_text

# Example candidate sentence
candidate = "There are 10 senior lecturers in the Department of Computational Mathematics."

# Load spaCy model for sentence tokenization and NER
nlp = spacy.load("en_core_web_trf")
sentences = [sent.text.strip() for sent in nlp(reference_text).sents if len(sent) > 1]

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract critical information like numbers using regex
def extract_critical_info(text):
    numbers = re.findall(r'\b\d+\b', text)
    return numbers

# Function to detect hallucinations
def detect_hallucinations(reference_text, candidate):
    # Step 1: Semantic Similarity
    candidate_scores = hallicinaware_bertscore.calculate_similarity(
        sentences=sentences, 
        candidates=[candidate]
    )
    highest_f1_score = np.max(candidate_scores)
    max_index = np.argmax(candidate_scores)
    most_similar_sentence = sentences[max_index]
    
    # Step 2: Factual Consistency
    reference_entities = extract_entities(reference_text)
    candidate_entities = extract_entities(candidate)
    hallucinated_entities = set(candidate_entities).difference(set(reference_entities))
    
    # Step 3: Exact Matching
    reference_info = extract_critical_info(reference_text)
    candidate_info = extract_critical_info(candidate)
    hallucinated_info = set(candidate_info).difference(set(reference_info))
    
    # Combine Results
    # Adjusting hallucination detection to focus on critical information
    hallucinations_detected = len(hallucinated_info) > 0
    
    return {
        "highest_f1_score": highest_f1_score,
        "most_similar_sentence": most_similar_sentence,
        "hallucinated_entities": hallucinated_entities,
        "hallucinated_info": hallucinated_info,
        "hallucinations_detected": hallucinations_detected
    }

# Detect hallucinations in the candidate sentence
result = detect_hallucinations(reference_text, candidate)

print(f"Highest F1 Score: {result['highest_f1_score']}")
print(f"Most similar sentence: {result['most_similar_sentence']}")
print(f"Hallucinated entities: {result['hallucinated_entities']}")
print(f"Hallucinated critical info: {result['hallucinated_info']}")
print(f"Hallucinations detected: {result['hallucinations_detected']}")

