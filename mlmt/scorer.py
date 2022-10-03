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
    ) -> float:

        mlm_score_list = []

        # 1つの文章をバッチとして処理
        for sentence in sentences:
            tokenized_dic = self.tokenizer(sentence, return_tensors='pt')
            input_ids = tokenized_dic["input_ids"]
            token_type_ids = tokenized_dic["token_type_ids"]
            attention_mask = tokenized_dic["attention_mask"]

            masked_attention_mask_list = []
            masked_input_ids_list = []
            sentence_size = input_ids.size(1)-1

            # input_idsとattention_maskに1つずらしでマスク処理
            for i in range(sentence_size):
                masked_input_ids = input_ids.clone()[0]
                masked_input_ids[i] = self.mask_id
                masked_input_ids_list.append(masked_input_ids)

                masked_attention_mask = attention_mask.clone()[0]
                masked_attention_mask[i] = 0
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


            log_likelihood = 0
            # 1トークンずつ予測確率を加算
            for out, masked_ids, ids in zip(logits, masked_input_ids_tensor, input_ids_tensor):
                target_index = masked_ids.tolist().index(self.mask_id)
                target_id = ids[target_index].item()
                if target_id == self.cls_id:
                    continue
                word_pred = out[target_index][target_id].item()
                log_likelihood += math.log(word_pred)
            # ave_log_likelihood = log_likelihood / sentence_size

            mlm_score_list.append(log_likelihood)

        return mlm_score_list