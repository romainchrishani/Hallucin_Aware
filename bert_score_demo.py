from hallucinaware.detection import HallucinAwareBERTScore
from hallucinaware import version
import spacy
from hallucinaware import utils

print("hallucin-aware version: ", version.__version__)

hallicinaware_bertscore = HallucinAwareBERTScore()
pdf_text = utils.read_pdf('resources/University2.pdf')
#pdf_text="I want to buy an iphone 13"
reference_text = pdf_text

candidate1="The Postgraduate Studies Division contributes to the Faculty of Information Technology by handling matters pertaining to taught postgraduate programmes and research degrees. This includes monitoring student progress, administering student feedback, and ensuring that students receive a high-quality education in their chosen fields of study. Additionally, the Postgraduate Studies Division collaborates with industry partners to develop new postgraduate programmes and research opportunities, which helps to keep the Faculty at the forefront of technological innovation and development. Overall, the Postgraduate Studies Division plays a crucial role in advancing knowledge and skills within the IT industry and preparing students for successful careers in this rapidly growing field."

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

