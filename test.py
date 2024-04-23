# from sentence_transformers import SentenceTransformer
from hallucinaware import utils
# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F

from hallucinaware.ngram import UnigramModel,NgramModel
# import spacy

# This is a sentence-transformers model: It maps sentences & paragraphs to a 
# 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
# model = SentenceTransformer('all-MiniLM-L6-v2')

# sentences = ["This is an example sentence.", "Another example sentence."]
# embeddings = model.encode(sentences)

# Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence: ", sentence)
#     print("Embedding: ", embedding)


# # Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# print(sentence_embeddings)

pdf_text = utils.read_pdf('./content/srilanka.pdf')

# ngram_model = NgramModel(3)
ngram_model = UnigramModel()

training_paragraph = pdf_text

ngram_model.add(training_paragraph)


main_response = "Sri Lanka is divided into 9 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Central, Southern, and Eastern Provinces mentioned earlier, the island nation comprises the Western, Northern, North Central, Uva, Sabaragamuwa, and North Western Provinces."
hallucinated_response = "India is divided into 28 states and 8 union territories, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the states of Uttar Pradesh, Maharashtra, and Tamil Nadu, which are known for their rich heritage and vibrant traditions, the country comprises the states of Rajasthan, Punjab, Himachal Pradesh, Uttarakhand, and Jammu and Kashmir in the north, renowned for their majestic mountains, desert landscapes, and historic forts. The eastern states of West Bengal, Odisha, Jharkhand, Bihar, and Assam boast diverse cultural influences, while the southern states of Andhra Pradesh, Telangana, Karnataka, and Kerala are famous for their temple towns, backwaters, and coastal charm. The central states of Madhya Pradesh, Chhattisgarh, and Gujarat showcase a unique amalgamation of architectural marvels and tribal cultures, while the western states of Goa and the union territories of Daman and Diu, and Dadra and Nagar Haveli offer a blend of Portuguese heritage and scenic beaches."
hallucinated_response2 = "Sri Lanka is divided into 3 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Eastern Province mentioned earlier, the island nation comprises the Western, and North Western Provinces."


# ngram_model.add(main_response)
# ngram_model.add(hallucinated_response)
ngram_model.add(hallucinated_response2)


ngram_model.train(k=0)

# evaluation_result = ngram_model.evaluate([main_response])
# print(evaluation_result)


# hallucinated_response_evaluation = ngram_model.evaluate([hallucinated_response])
# print(hallucinated_response_evaluation)

hallucinated_response_evaluation2 = ngram_model.evaluate([hallucinated_response2])
print(hallucinated_response_evaluation2)

# Average Negative Log Probability: Lower values indicate that the response is more likely according to the model.
# Maximum Negative Log Probability: A higher value indicates that at least one token in the response is less likely according to the model.