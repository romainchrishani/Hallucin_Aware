import spacy
import bert_score
import numpy as np
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AutoModelForCausalLM
from hallucinaware.utils import *
from hallucinaware.ngram import UnigramModel, NgramModel
import torch
from torch.nn import CrossEntropyLoss
from tqdm import auto as tqdm_lib
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
from transformers import pipeline

class HallucinAwareBERTScoreV1:
    def __init__(self):
        #self.nlp = spacy.load("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_trf")
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

            reference_expanded  = expand_list(sentences, num_sentences_candidate)
            sample_expanded = duplicate_sentences(candidate_sentences, num_sentences)

            P, R, F1 = bert_score.score(
                            sample_expanded, reference_expanded,
                            lang="en", verbose=False,
                            rescale_with_baseline=True,
                        )

            F1_score_matrix = F1.reshape(num_sentences, num_sentences_candidate)
                
            F1_arr_max_axis1 = F1_score_matrix.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()
            
            bertscore_array[:,c] = F1_arr_max_axis1
        
        bertscore_mean = bertscore_array.mean(axis=-1)

        one_minus_bertscore_mean = 1.0 - bertscore_mean

        return one_minus_bertscore_mean

#my working one
class HallucinAwareBERTScore:
    def __init__(self):
        #self.nlp = spacy.load("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_trf")
        print("initializing ModifiedHallucinAwareBERTScore...")

    def calculate_similarity(self, sentences: list, candidates: list):
        num_sentences = len(sentences)
        num_candidates = len(candidates)
        bertscore_array = np.zeros((num_sentences, num_candidates))

        for c in range(num_candidates):
            candidate = candidates[c]
            candidate_sentences = [sent.text.strip() for sent in self.nlp(candidate).sents if len(sent) > 0]
            num_sentences_candidate = len(candidate_sentences)

            reference_expanded = sentences * num_sentences_candidate
            sample_expanded = candidate_sentences * num_sentences

            P, R, F1 = bert_score.score(
                sample_expanded, reference_expanded,
                lang="en", verbose=False,
                rescale_with_baseline=True,
            )

            F1_score_matrix = F1.reshape(num_sentences, num_sentences_candidate)
            F1_arr_max_axis1 = F1_score_matrix.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:, c] = F1_arr_max_axis1

        bertscore_mean = bertscore_array.mean(axis=-1)
        return bertscore_mean

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
    
