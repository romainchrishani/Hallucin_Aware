from hallucinaware import utils
from hallucinaware.detection import HallucinAwareNgram

pdf_text = utils.read_pdf('./content/srilanka.pdf')

ngram_model = HallucinAwareNgram(2)

training_paragraph = pdf_text

main_response = "Sri Lanka is divided into 9 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Central, Southern, and Eastern Provinces mentioned earlier, the island nation comprises the Western, Northern, North Central, Uva, Sabaragamuwa, and North Western Provinces."
hallucinated_response = "India is divided into 28 states and 8 union territories, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the states of Uttar Pradesh, Maharashtra, and Tamil Nadu, which are known for their rich heritage and vibrant traditions, the country comprises the states of Rajasthan, Punjab, Himachal Pradesh, Uttarakhand, and Jammu and Kashmir in the north, renowned for their majestic mountains, desert landscapes, and historic forts. The eastern states of West Bengal, Odisha, Jharkhand, Bihar, and Assam boast diverse cultural influences, while the southern states of Andhra Pradesh, Telangana, Karnataka, and Kerala are famous for their temple towns, backwaters, and coastal charm. The central states of Madhya Pradesh, Chhattisgarh, and Gujarat showcase a unique amalgamation of architectural marvels and tribal cultures, while the western states of Goa and the union territories of Daman and Diu, and Dadra and Nagar Haveli offer a blend of Portuguese heritage and scenic beaches."
hallucinated_response2 = "Sri Lanka is divided into 3 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Eastern Province mentioned earlier, the island nation comprises the Western, and North Western Provinces."


res = ngram_model.calculate( 0.5, [hallucinated_response2], pdf_text, [main_response,hallucinated_response,hallucinated_response2])


print(res)

# Average Negative Log Probability: Lower values indicate that the response is more likely according to the model.
# Maximum Negative Log Probability: A higher value indicates that at least one token in the response is less likely according to the model.



# 