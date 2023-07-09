import math
import transformers
from transformers import  AutoTokenizer
import torch

from .model import AutoMLModel

class MLMScorer():
    def __init__(
        self,
        pretrained_model_name: str,
        model_config: transformers.AutoConfig=None,
        use_cuda: bool=False
    ):
        if use_cuda:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                print('CUDA is not available on your device. So run CPU.')
                self.device = 'cpu'
            self.model = AutoMLModel(pretrained_model_name, model_config).to(self.device)
        else:
            self.device = 'cpu'
            self.model = AutoMLModel(pretrained_model_name, model_config).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

    def score_sentences(
        self,
        sentences: str,
        get_token_likelihood: bool=False,
    ) -> float:

        if get_token_likelihood:
            mlm_score_list = [{}]*len(sentences)
        else:
            mlm_score_list = [""]*len(sentences)

        # 1つの文章をバッチとして処理
        for i, sentence in enumerate(sentences):
            tokenized_dic = self.tokenizer(sentence, return_tensors='pt')
            input_ids = tokenized_dic["input_ids"]
            token_type_ids = tokenized_dic["token_type_ids"]
            attention_mask = tokenized_dic["attention_mask"]

            masked_attention_mask_list = []
            masked_input_ids_list = []
            sentence_size = input_ids.size(1)-1

            # input_idsとattention_maskに1つずらしでマスク処理
            for j in range(sentence_size):
                masked_input_ids = input_ids.clone()[0]
                masked_input_ids[j] = self.mask_id
                masked_input_ids_list.append(masked_input_ids)

                masked_attention_mask = attention_mask.clone()[0]
                masked_attention_mask[j] = 0
                masked_attention_mask_list.append(masked_attention_mask)

            masked_input_ids_tensor = torch.stack(masked_input_ids_list)[1:]
            token_type_ids_tensor = token_type_ids.expand(sentence_size, sentence_size+1)[1:]
            masked_attention_mask_tensor = torch.stack(masked_attention_mask_list)[1:]
            input_ids_tensor = input_ids.expand(sentence_size, sentence_size+1)[1:]


            with torch.no_grad():
                logits = self.model(
                    input_ids=masked_input_ids_tensor.to(self.device),
                    token_type_ids=token_type_ids_tensor.to(self.device),
                    attention_mask=masked_attention_mask_tensor.to(self.device)
                )

            log_likelihood_sum = 0
            token_prob_list = [0]*len(logits)
            # 1トークンずつ予測確率を加算
            for j, out, masked_ids, ids in zip(range(len(logits)), logits, masked_input_ids_tensor, input_ids_tensor):
                target_index = masked_ids.tolist().index(self.mask_id)
                target_id = ids[target_index].item()
                word_pred = out[target_index][target_id].item()
                # print(word_pred)
                log_likelihood = math.log(word_pred)
                token_prob_list[j] = log_likelihood
                log_likelihood_sum += log_likelihood
            # ave_log_likelihood = log_likelihood / sentence_size

                if get_token_likelihood:
                    mlm_score_list[i] = {
                        "all": log_likelihood_sum,
                        "token": token_prob_list,
                    }
                else:
                    mlm_score_list[i] = log_likelihood_sum

        return mlm_score_list