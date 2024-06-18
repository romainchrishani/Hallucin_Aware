import spacy
import bert_score
import numpy as np
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AutoModelForCausalLM
from hallucinaware.utils import *
from hallucinaware.ngram import UnigramModel, NgramModel
import torch
from torch.nn import CrossEntropyLoss
from tqdm import auto as tqdm_lib


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
        scores = np.zeros((sentence_count, candidate_count))
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
                prob_ = probs[0][1].item() # prob(contradiction)
                scores[_s, _c] = prob_
        sentence_wise_score = scores.mean(axis=-1)
        return sentence_wise_score        
    
    

class HallucinAwarePerplexity:
    def __init__(self,
            device = None,
            model_name: str = None
        ):

        self.model_name = model_name if model_name is not None else "gpt2"
        if device is not None:
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"       
                
        print(f"initializing HallucinAwarePerplexity with {device}...")   

        
    def calculate(
        self,
        data,
        batch_size: int = 16,
        add_start_token: bool = True,
        device=None, max_length=None
    ):    
        print("calculate --> ")
        model = AutoModelForCausalLM.from_pretrained(self.model_name)    
        model = model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if tokenizer.pad_token is None and batch_size > 1:
            model_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            print("model_special_tokens: ", model_special_tokens)
            assert(len(model_special_tokens) > 0), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            tokenizer.add_special_tokens({"pad_token": model_special_tokens[0]})
            
        if add_start_token and max_length:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length    

        encodings = tokenizer(
            data,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]
        
        print(encodings)
        
        if add_start_token:
                assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
        
        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")
        
        for start_index in tqdm_lib.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}          