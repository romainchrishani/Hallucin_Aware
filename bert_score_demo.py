#working one, one by one
from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils
import numpy as np
import bert_score

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()
pdf_text = utils.read_pdf('resources/2023-All.pdf')

reference_text = pdf_text

candidate="The faculty of Information Technology was established in June 2001."

nlp = spacy.load("en_core_web_trf")
sentences = [sent for sent in nlp(reference_text).sents] 
reference_sentences = [sent.text.strip() for sent in sentences if len(sent) > 1]

candidate_scores = hallicinaware_bertscore.calculate_similarity(
    sentences=reference_sentences, 
    candidates=[candidate]
)
max_index = np.argmax(candidate_scores)
most_similar_sentence = reference_sentences[max_index]
highest_f1_score = candidate_scores[max_index]

print(f"Most similar sentence: {most_similar_sentence}")
print(f"Highest F1 Score: { highest_f1_score}")  
print(f"Highest BERTScore F1: {1.0 - highest_f1_score}")  

