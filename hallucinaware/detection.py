import spacy
import bert_score
import numpy as np
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from hallucinaware.utils import *
from hallucinaware.ngram import UnigramModel, NgramModel
import torch

class HallucinAwareBERTScore:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        print("initializing HallucinAwareBERTScore...")

    def calculate_similarity(
            self, 
            sentences: list, 
            candidates: list
        ):

        num_sentences = len(sentences)
        num_candidates = len(candidates)
        bertscore_array = np.zeros((num_sentences, num_candidates))
        
        for c in range(num_candidates):
            candidate = candidates[c]
            candidate_sentences  = [sent.text.strip() for sent in self.nlp(candidate).sents if len(sent) > 0]
            num_sentences_candidate = len(candidate_sentences)
            
            # print("\ncandidate_sentences: \n" + '\n'.join(candidate_sentences))

            reference_expanded  = expand_list(sentences, num_sentences_candidate)
            sample_expanded = duplicate_sentences(candidate_sentences, num_sentences)

            # print("\nreference_expanded:", reference_expanded)
            # print("\nsample_expanded :", sample_expanded)

            P, R, F1 = bert_score.score(
                            sample_expanded, reference_expanded,
                            lang="en", verbose=False,
                            rescale_with_baseline=True,
                        )
        
            # print("\nF1 score:", F1)

            F1_score_matrix = F1.reshape(num_sentences, num_sentences_candidate)
        
            # print("\nF1_score_matrix:\n", F1_score_matrix)
        
            F1_arr_max_axis1 = F1_score_matrix.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()
            
            # print("\nF1_arr_max_axis1:", F1_arr_max_axis1)

            bertscore_array[:,c] = F1_arr_max_axis1
            # print("\nbertscore_array:\n", bertscore_array)
            # print("===========================")
        

        bertscore_mean = bertscore_array.mean(axis=-1)
        # print("\nmean bertscore per sentence:\n", bertscore_mean)

        one_minus_bertscore_mean = 1.0 - bertscore_mean
        #print("\1 - mean bertscore per sentence:\n", one_minus_bertscore_mean)

        return one_minus_bertscore_mean

class HallucinAwareNgram:
    def __init__(self, n: int):
        self.n = n #n=1 is Unigram, n=2 is Bigram, etc.
        print(f"initializing HallucinAwareNgram with {n}-gram...")
    
    def calculate(
        self,
        k,
        sentences: list,
        original_text: str,
        generated_text: list
    ):
        if self.n == 1:
            ngram_model = UnigramModel()
        elif self.n > 1:
            ngram_model = NgramModel(self.n) 
        else:
            raise ValueError("n must be integer >= 1")
        
        ngram_model.add(original_text)
        for candidate in generated_text:
            ngram_model.add(candidate)
        ngram_model.train(k=k)
        ngram_pred = ngram_model.evaluate(sentences)
        return ngram_pred    
    
class HallucinaAwareDeBERTa:
    def __init__ (
        self, 
        model_name: str = None, 
        device = None
    ):
            print(f"initializing HallucinaAwareDeBERTa with {device}...")
            model_name = model_name if model_name is not None else "microsoft/deberta-v3-large"
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            if device is None:
                device = torch.device("cpu")
            self.model.to(device)
            self.device = device    

    
    def calculate(self,
        sentences: list,
        candidates: list
    ):
        sentence_count = len(sentences)
        candidate_count = len(candidates)
        scores = np.zeros(sentence_count, candidate_count)
        for _s, sentence in enumerate(sentences):
            for _c, candidate in enumerate(candidates):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, candidate)],
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][1].item()
                scores[_s, _c] = prob_
        sentence_wise_score = scores.mean(axis=-1)
        return sentence_wise_score        
                