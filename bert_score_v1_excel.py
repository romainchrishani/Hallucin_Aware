#first one excel
import pandas as pd
from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils

print("hallucin-aware version: ", version.__version__)

# Initialize HallucinAwareBERTScore
hallucinaware_bertscore = HallucinAwareBERTScore()

# Load the PDF text
pdf_text = utils.read_pdf('resources/2023-5.pdf')
reference_text = pdf_text

# Load SpaCy model and extract sentences from reference text
nlp = spacy.load("en_core_web_sm")
sentences = [sent for sent in nlp(reference_text).sents]
reference_sentences = [sent.text.strip() for sent in sentences if len(sent) > 1]

# Read candidates from Excel file
candidates_df = pd.read_excel('resources/Candidates.xlsx')
candidates = candidates_df['Candidate'].tolist()  # Assuming the column name is 'Candidate'

# Prepare a list to store results
results = []

# Calculate BERTScore for each candidate
for candidate in candidates:
    candidate_scores = hallucinaware_bertscore.calculate_similarity(
        sentences=reference_sentences,
        candidates=[candidate]
    )
# Check for hallucination based on threshold
    threshold = 0.001
    for score in candidate_scores:
        if score < threshold:
            results.append((candidate, score, "Response seems coherent."))
            break
    else:
        results.append((candidate, score, "Potential hallucination detected!"))


# Create a DataFrame to save results
results_df = pd.DataFrame(results, columns=['Candidate', 'BERTScore', 'Evaluation'])

# Save the results to a new Excel file
results_df.to_excel('resources/resultsv1.xlsx', index=False)

print("Results have been saved to 'resources/results.xlsx'")
