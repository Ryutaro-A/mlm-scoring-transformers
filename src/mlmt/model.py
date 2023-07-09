import transformers
from transformers import AutoModelForMaskedLM, AutoConfig
import torch
from torch import nn
from torch.nn import functional as F


class AutoMLModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        config: transformers.AutoConfig=None
    ):
        super(AutoMLModel, self).__init__()
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
        
        self.bert_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name, config=config)


    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        output = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        logits = output.logits

        # 全結合層からの出力に対してsoftmaxを計算
        return F.softmax(logits, dim=1)

