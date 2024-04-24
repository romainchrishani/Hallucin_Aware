import pandas as pd
from hallucinaware.ngram import NgramModel
from hallucinaware.utils import read_pdf

excel_path = 'resources/Answers for University PDF2.xlsx' 
df = pd.read_excel(excel_path)

responses = df.iloc[:, 0].tolist()

ngram_model = NgramModel(2)

pdf_text = read_pdf('resources/University2.pdf')  
ngram_model.add(pdf_text)
ngram_model.train(k=0.5)  

results = []

for response in responses:
    ngram_model.add(response)
    ngram_model.train(k=0.5)  
    evaluation_result = ngram_model.evaluate([response])

    result_entry = {
        "Response": response,
        "Sentence Level Avg": evaluation_result['sentence_level']['negative_log_prob_avg'],
        "Sentence Level Max": evaluation_result['sentence_level']['negative_log_prob_max'],
        "Document Level Avg": evaluation_result['document_level']['negative_log_prob_avg'],
        "Document Level Max": evaluation_result['document_level']['negative_log_prob_max']
    }
    results.append(result_entry)

results_df = pd.DataFrame(results)
results_df.to_excel('evaluation_results.xlsx', index=False)


# Average Negative Log Probability: Lower values indicate that the response is more likely according to the model.
# Maximum Negative Log Probability: A higher value indicates that at least one token in the response is less likely according to the model.



# ngram_model2 = HallucinAwareNgram(2)