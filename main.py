from hallucinaware.detection import HallucinAwareVectorBERTScore
from hallucinaware import version
from hallucinaware import utils
import spacy

print("hallucin-aware version: ", version.__version__)




pdf_text = utils.read_pdf('./content/srilanka.pdf')

print("pdf_text: ", pdf_text)

document_embeddings = utils.encode_sentences([pdf_text])

index = utils.create_embedding_index(document_embeddings)

hallicinaware_bertscore = HallucinAwareVectorBERTScore(index)

candidate1 = "Sri Lanka is divided into 9 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Central, Southern, and Eastern Provinces mentioned earlier, the island nation comprises the Western, Northern, North Central, Uva, Sabaragamuwa, and North Western Provinces."
candidate2 = "India is divided into 28 states and 8 union territories, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the states of Uttar Pradesh, Maharashtra, and Tamil Nadu, which are known for their rich heritage and vibrant traditions, the country comprises the states of Rajasthan, Punjab, Himachal Pradesh, Uttarakhand, and Jammu and Kashmir in the north, renowned for their majestic mountains, desert landscapes, and historic forts. The eastern states of West Bengal, Odisha, Jharkhand, Bihar, and Assam boast diverse cultural influences, while the southern states of Andhra Pradesh, Telangana, Karnataka, and Kerala are famous for their temple towns, backwaters, and coastal charm. The central states of Madhya Pradesh, Chhattisgarh, and Gujarat showcase a unique amalgamation of architectural marvels and tribal cultures, while the western states of Goa and the union territories of Daman and Diu, and Dadra and Nagar Haveli offer a blend of Portuguese heritage and scenic beaches."
candidate3 = "Sri Lanka is divided into 3 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Eastern Province mentioned earlier, the island nation comprises the Western, and North Western Provinces."

candidate_scores = hallicinaware_bertscore.calculate_similarity([candidate1, candidate2, candidate3])


print("\nBERTScore")
for s1 in candidate_scores:
    print("{:.4f}".format(s1))

# qa_pipeline = utils.setup_qa_system()

# question = "How many provinces does sri lanka have?"

# query_embedding = utils.encode_sentences([question])

# retrieved_docs = utils.retrieve_similar_documents(query_embedding, index, [pdf_text])

# print(retrieved_docs)

# answer = utils.answer_question_with_qa_system(question, retrieved_docs[0], qa_pipeline)
# print("answer: ")
# print(answer)


# nlp = spacy.load("en_core_web_sm")
# sentences = [sent for sent in nlp(pdf_text).sents] 
# sentences = [sent.text.strip() for sent in sentences if len(sent) > 0]

# print("\n\nsentences: \n", '\n'.join(sentences))

# print("\nrunning on {} sentences...".format(len(sentences)))

# answer2 = '''Sri Lanka is divided into 9 provinces.'''.replace("\n", " ").strip()

# similarity_score = hallicinaware_bertscore.calculate_similarity(sentences, [answer])


# print("\nBERTScore")
# for s1 in similarity_score:
#     print("{:.4f}".format(s1))




# answer = '''Sri Lanka is divided into 10 provinces.'''.replace("\n", " ").strip()

# similarity_score = hallicin_aware.calculate_similarity_with_bertscore(sentences, answer)

# for s1, s2 in similarity_score:
    # print("{:.4f}\t{:.4f}".format(s1, s2))
    