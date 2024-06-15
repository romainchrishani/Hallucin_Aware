import pandas as pd
from hallucinaware.ngram import UnigramModel
from hallucinaware.utils import read_pdf  
import copy 

excel_path = 'resources/1.xlsx' 
df = pd.read_excel(excel_path)

responses = df.iloc[:, 0].tolist()

ngram_model = UnigramModel()

pdf_text = read_pdf('resources/University2.pdf')  
ngram_model.add(pdf_text)

results = []

for response in responses:
    ngram_model_copy=copy.deepcopy(ngram_model)
    ngram_model_copy.add(response)
    ngram_model_copy.train(k=0)  
    evaluation_result = ngram_model_copy.evaluate([response])

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