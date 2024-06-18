import spacy
import numpy as np
from nltk.util import ngrams
from collections import defaultdict
import math
from collections import Counter

class UnigramModel:
    """
        Unigram Model Equation:

    P(w1, w2, ..., wn) = ∏_{i=1}^{n} P(wi)

    Where:
    - P(wi) is the probability of the i-th word in the document.
    - w1, w2, ..., wn are the words in the document.
    - n is the total number of words in the document.
    - P(wi) = Count(wi) / Total number of words in the corpus
    """

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.num_sentences = 0
        self.num_tokens = 0
        self.unigram_counts = {"<unknown>": 0}

    def add(self, text: str) -> None:
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            # print(sentence)
            tokens = [token.text for token in self.nlp(sentence)]
            tokens = [token.lower() for token in tokens]
            # print(tokens)
            self.num_sentences += 1
            self.num_tokens += len(tokens)

            for token in tokens:
                if token not in self.unigram_counts:
                    self.unigram_counts[token] = 1
                else:
                    self.unigram_counts[token] += 1
        # print(self.unigram_counts)

    def train(self, k: int = 0) -> None:
        self.probs = {}
        for unigram, unigram_count in self.unigram_counts.items():
            prob_nume = unigram_count + k
            prob_denom = self.num_tokens + k * len(self.unigram_counts)
            self.probs[unigram] = prob_nume / prob_denom
        # print(self.probs)

    def evaluate(self, sentences: list) -> float:
        negative_log_prob_avg = []  # Negative Log Probability
        negative_log_prob_max = []
        doc_negative_log_prob_avg = []
        doc_negative_log_prob_max = []
        doc_log_prob = []
        
        for sentence in sentences:
            sentence_log_prob = []
            tokens = [token.text for token in self.nlp(sentence)]
            # print(tokens)  
            
            for token in tokens:
                token = token.lower()
                if token not in self.unigram_counts:
                    token = '<unknown>'
                train_prob = self.probs[token]
                logprob = np.log(train_prob)
                sentence_log_prob.append(logprob)
                doc_log_prob.append(logprob)
                
            # print("sentence_log_prob: ", sentence_log_prob)    
            # print("doc_log_prob: ", doc_log_prob)  
            
            negative_log_prob_avg += [-1.0 * np.mean(sentence_log_prob)]
            negative_log_prob_max += [-1.0 * np.min(sentence_log_prob)]    
            
            # print("negative_log_prob_avg: ", negative_log_prob_avg)
            # print("negative_log_prob_max: ", negative_log_prob_max)
            # print("\n")
            
        doc_negative_log_prob_avg = -1.0 * np.mean(doc_log_prob)
        doc_negative_log_prob_max = np.mean(negative_log_prob_max)    
        
        # print("doc_negative_log_prob_avg: ", doc_negative_log_prob_avg)
        # print("doc_negative_log_prob_max: ", doc_negative_log_prob_max)
        # print("\n")
            
        return {
            'sentence_level': {'negative_log_prob_avg': negative_log_prob_avg, 'negative_log_prob_max': negative_log_prob_max},
            'document_level': {'negative_log_prob_avg': doc_negative_log_prob_avg, 'negative_log_prob_max': doc_negative_log_prob_max},
        }

class NgramModel:
    def __init__(
        self, n: int, left_pad_symbol: str = '<s>') -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_count = 0
        self.ngram_count = 0
        self.counts = {'<unknown>': 0}
        self.n = n
        self.left_pad_symbol = left_pad_symbol

    def add(self, text: str) -> None:
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            tokens = [token.lower() for token in tokens]
            ngs = list(ngrams(tokens, n=self.n, pad_left=True, left_pad_symbol=self.left_pad_symbol))
            assert len(ngs) == len(tokens)
            self.sentence_count += 1
            self.ngram_count += len(ngs)
            for ng in ngs:
                if ng not in self.counts:
                    self.counts[ng] = 1
                else:
                    self.counts[ng] += 1

    def train(self, k: int = 0) -> None:
        self.probs = {}
        for ngram, ngram_count in self.counts.items():
            prob_nom = ngram_count + k
            prob_denom = self.ngram_count + k * len(self.counts) 
            self.probs[ngram] = prob_nom / prob_denom

    def evaluate(self, sentences: list) -> float:
        negative_log_prob_avg = []
        negative_log_prob_max = []
        doc_log_prob = []
        for sentence in sentences:
            logprob_sent = []
            tokens = [token.text for token in self.nlp(sentence)]
            tokens_ = [tok.lower() for tok in tokens]
            ngs = list(ngrams(tokens_, n=self.n, pad_left=True, left_pad_symbol=self.left_pad_symbol))
            assert len(ngs) == len(tokens)
            for token, ng in zip(tokens, ngs):
                if ng not in self.counts:
                    ng = '<unknown>'
                train_prob = self.probs[ng]
                logprob = np.log(train_prob)
                logprob_sent.append(logprob)
                doc_log_prob.append(logprob)
            negative_log_prob_avg += [-1.0 * np.mean(logprob_sent)]
            negative_log_prob_max += [-1.0 * np.min(logprob_sent)]
        doc_negative_log_prob_avg = -1.0 * np.mean(doc_log_prob)
        doc_negative_log_prob_max = np.mean(negative_log_prob_max)
        return {
            'sentence_level': {'negative_log_prob_avg': negative_log_prob_avg, 'negative_log_prob_max': negative_log_prob_max},
            'document_level': {'negative_log_prob_avg': doc_negative_log_prob_avg, 'negative_log_prob_max': doc_negative_log_prob_max},
        }


