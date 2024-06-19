from hallucinaware import utils
from hallucinaware.ngram import NgramModel
from hallucinaware.detection import HallucinAwareNgram

pdf_text = utils.read_pdf('resources/2023.pdf')
#pdf_text="Sri Lanka is divided into 9 provinces, each offering a distinct blend of attractions, culture, and natural beauty. In addition to the Central, Southern, and Eastern Provinces mentioned earlier, the island nation comprises the Western, Northern, North Central, Uva, Sabaragamuwa, and North Western Provinces."

ngram_model = NgramModel(3)

training_paragraph = pdf_text

ngram_model.add(training_paragraph)

main_response="The student should submit a Medical Certificate from a registered Medical Practitioner supported by other documents mentioned in step 2 or 4 to the Senior Assistant Registrar Examinations.Question: What should a student do if they miss more than one exam paper or the whole exam?Helpful Answer: The student should submit the following documents with the duly completed application form for consideration of absence from the examination on medical ground: a proof of informing the Senior Assistant Registrar Examinationsa Medical Certificate issued by a Government Hospital supported by the diagnosis card receipt of payment for the medical certificateresults of investigations carried out such as blood test etc if any and prescription forms for the drug treatment givenIf no diagnosis card is available, other documents mentioned above are essential.Question: If a student misses only one paper, what should they do?Helpful Answer: The student may submit a Medical Certificate from a registered Medical Practitioner supported by other documents mentioned in step 2 or 4.Question: If a student is currently undergoing medical treatment for a mental disorder, what should they do?Helpful Answer: The student should inform the University Medical Officer before the commencement of the examination, and in case of a student developing a mental disorder during the term time or during the examination, they are strongly advised to seek advice and treatment from the University Medical Officer. In the event of the student taking treatment from external sources, they should submit a medical certificate to the UMO within 48 hours."

ngram_model.add(main_response)

ngram_model.train(k=0.5)

evaluation_result = ngram_model.evaluate([main_response])
print(evaluation_result)


# Average Negative Log Probability: Lower values indicate that the response is more likely according to the model.
# Maximum Negative Log Probability: A higher value indicates that at least one token in the response is less likely according to the model.


