from hallucinaware import utils

pdf_text = utils.read_pdf('./content/srilanka.pdf')
print("pdf_text: ", pdf_text)

document_embeddings = utils.encode_sentences([pdf_text])
# print("document_embeddings: \n", document_embeddings)

index = utils.create_embedding_index(document_embeddings)
# print("index dimensionality: ", index.d)


question = "What is the UNESCO World Heritage in Sri Lanka?"

query_embedding = utils.encode_sentences([question])
# print("query_embedding: \n", query_embedding)

retrieved_docs = utils.retrieve_similar_documents(query_embedding, index, [pdf_text])
print("retrieved_docs: \n", retrieved_docs)

qa_pipeline = utils.setup_qa_system()

answers = utils.answer_question_with_qa_system(question, retrieved_docs[0], qa_pipeline)

print("question: ", question)
print("answer: ", answers[0]['answer'])
print("answer: ", answers[1]['answer'])
print("answer: ", answers[2]['answer'])
print("answer: ", answers[3]['answer'])
print("answer: ", answers[4]['answer'])

