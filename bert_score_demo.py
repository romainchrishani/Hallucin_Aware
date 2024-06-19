'''from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()
pdf_text = utils.read_pdf('resources/2023.pdf')
#pdf_text="I want to buy an iphone 13"
reference_text = pdf_text

candidate1="The question is not clear, maybe you can ask for specific details about the type of examination."
nlp = spacy.load("en_core_web_sm")
sentences = [sent for sent in nlp(reference_text).sents] 
reference_sentences  = [sent.text.strip() for sent in sentences if len(sent) > 1]

candidate_scores = hallicinaware_bertscore.calculate_similarity(
    sentences = reference_sentences, 
    candidates = [candidate1] # input generated answers
    )


print("\nBERTScore")
#for s1 in candidate_scores:
#   print("{:.4f}".format(s1))


threshold = 0.001

# Check for hallucination
for score in candidate_scores:
    if score < threshold:
        print("Candidate BERTScore: ",score)
        print("\nResponse seems coherent.")
        break 
else:
    print("Candidate BERTScore: ",score)
    print("\nPotential hallucination detected!")
'''

from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils
import numpy as np
import bert_score

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()
pdf_text = utils.read_pdf('resources/University2.pdf')

reference_text = pdf_text

# Example candidate sentence
candidate_sentence = "The Faculty of Information Technology was established in June 2001, so it is currently around 19 years old (as of October 2020)."

# Load spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")
sentences = [sent for sent in nlp(reference_text).sents] 
reference_sentences = [sent.text.strip() for sent in sentences if len(sent) > 1]

# Calculate BERTScore similarity
candidate_scores = hallicinaware_bertscore.calculate_similarity(
    sentences=reference_sentences, 
    candidates=[candidate_sentence]
)

# Find the sentence with the highest BERTScore
max_index = np.argmax(candidate_scores)
most_similar_sentence = reference_sentences[max_index]
highest_f1_score = candidate_scores[max_index]

print(f"Most similar sentence: {most_similar_sentence}")
print(f"Highest BERTScore F1: {1.0 - highest_f1_score}")  # since scores were returned as 1 - BERTScore
