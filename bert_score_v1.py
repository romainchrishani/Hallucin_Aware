#first one
from hallucinaware.detection import HallucinAwareBERTScoreV1
from hallucinaware import version
import spacy
from hallucinaware import utils

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScoreV1()
pdf_text = utils.read_pdf('resources/2023.pdf')
#pdf_text="I want to buy an iphone 13"
reference_text = pdf_text

candidate1="The Department’s dedicated staff comprises eight doctorly qualified and well-renowned experts in AI, Mathematics and Statistics, including one Senior Professor, 10 Senior Lecturers, one Junior Lecturer, three instructors, and five non-academic staff members."
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
