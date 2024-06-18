from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils
import numpy as np
import pandas as pd

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()

pdf_text = utils.read_pdf('resources/University2.pdf')

reference_text = pdf_text

nlp = spacy.load("en_core_web_sm")
sentences = [sent.text.strip() for sent in nlp(reference_text).sents if len(sent) > 1]
reference_sentences = sentences

# Read candidate sentences from an Excel sheet
input_file = 'resources/BertScore.xlsx'
output_file = 'candidate_scores.xlsx'

df = pd.read_excel(input_file)
candidates = df['CandidateSentences'].tolist()

# Initialize a list to store results
results = []

# Calculate BERTScore for each candidate sentence
for candidate in candidates:
    candidate_scores = hallicinaware_bertscore.calculate_similarity(
        sentences=reference_sentences,
        candidates=[candidate]
    )

    # Save the candidate sentence and its score
    for score in candidate_scores:
        results.append({'CandidateSentence': candidate, 'BERTScore': score})

# Create a DataFrame to save the results
results_df = pd.DataFrame(results)

# Save the results to a new Excel sheet
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")