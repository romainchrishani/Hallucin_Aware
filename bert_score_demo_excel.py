#working one excel
from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils
import numpy as np
import pandas as pd

print("hallucin-aware version: ", version.__version__)

# Initialize HallucinAwareBERTScore
hallucinaware_bertscore = HallucinAwareBERTScore()

# Read PDF text
pdf_text = utils.read_pdf('resources/2023-5.pdf')
reference_text = pdf_text

# Load spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")
sentences = [sent for sent in nlp(reference_text).sents] 
reference_sentences = [sent.text.strip() for sent in sentences if len(sent) > 1]

# Read candidates from Excel file
candidates_df = pd.read_excel('resources/Candidate1.xlsx')
candidates = candidates_df['Candidate'].tolist()  # Assuming the column name is 'Candidate'

# Prepare a list to store results
results = []

# Calculate BERTScore for each candidate
for candidate in candidates:
    candidate_scores = hallucinaware_bertscore.calculate_similarity(
        sentences=reference_sentences,
        candidates=[candidate]
    )

    # Find the sentence with the highest BERTScore
    max_index = np.argmax(candidate_scores)
    most_similar_sentence = reference_sentences[max_index]
    highest_f1_score = candidate_scores[max_index]
    final_score = (1.0 - highest_f1_score)

    results.append((candidate, final_score, highest_f1_score, most_similar_sentence))

# Create a DataFrame from results and save to a new Excel file
results_df = pd.DataFrame(results, columns=['Candidate', 'final_score', 'highest_f1_score', 'most_similar_sentence'])
results_df.to_excel('resources/results.xlsx', index=False)

print("Results have been saved to 'resources/results.xlsx'")
