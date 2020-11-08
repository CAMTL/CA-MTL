import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from src.model.decoder import Decoder
from src.model.encoders.bert import _BertEncoder
from src.model.encoders.ca_mtl_base import CaMtlBaseEncoder
from src.model.encoders.ca_mtl_large import CaMtlLargeEncoder

logger = logging.getLogger(__name__)


@dataclass
class CaMtlArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from: CA-MTL-base, CA-MTL-large, bert-base-cased "
                    "bert-base-uncased, bert-large-cased, bert-large-uncased"
        }
    )


class CaMtl(BertPreTrainedModel):
    def __init__(
        self,
        config,
        model_args,
        data_args,
    ):
        super().__init__(config)

        self.data_args = data_args
        self.bert = self._create_encoder(model_args.model_name_or_path)
        self.decoders = nn.ModuleList()
        for task in data_args.tasks:
            self.decoders.append(Decoder(config.hidden_size, task))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_id=None,
        span_locs=None,
        sample_id=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            task_id=task_id,
        )

        sequence_output, pooled_output = outputs[:2]

        loss_list = []
        unique_task_ids = torch.unique(task_id)
        unique_task_ids_list = (
            unique_task_ids.cpu().numpy()
            if unique_task_ids.is_cuda
            else unique_task_ids.numpy()
        )
        loss_grouped_per_task = (
            torch.zeros_like(task_id[0]).repeat(len(self.data_args.tasks)).float()
        )
        batch_entropy_per_task = torch.zeros(input_ids.shape[0])
        batch_entropy_mean_per_task = torch.zeros(input_ids.shape[0])
        max_mean_batch_entropy = None
        logits = None
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_id == unique_task_id
            decoder_id = unique_task_id
            logits, current_loss, batch_entropy = self.decoders[decoder_id].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            batch_entropy_mean = batch_entropy.mean().item()
            batch_entropy_per_task[task_id_filter] = batch_entropy
            batch_entropy_mean_per_task[task_id_filter] = torch.full_like(
                batch_entropy, batch_entropy_mean
            )
            if (
                max_mean_batch_entropy is None
                or batch_entropy_mean > max_mean_batch_entropy
            ):
                max_mean_batch_entropy = batch_entropy_mean

            if labels is not None:
                loss_grouped_per_task[unique_task_id] = current_loss
                loss_list.append(current_loss)

        outputs = (
            (logits,)
            + outputs[2:]
            + (
                batch_entropy_per_task,
                batch_entropy_mean_per_task,
                max_mean_batch_entropy,
            )
        )

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs + (loss_grouped_per_task.view(1, -1),)

        return outputs

    def _create_encoder(self, model_name_or_path):
        if model_name_or_path == "CA-MTL-large":
            return CaMtlLargeEncoder(self.config, data_args=self.data_args)
        elif model_name_or_path == "CA-MTL-base":
            return CaMtlBaseEncoder(self.config, data_args=self.data_args)
        else:
            return _BertEncoder(self.config)

    @staticmethod
    def get_base_model(model_name_or_path):
        if model_name_or_path == "CA-MTL-large":
            return "bert-large-cased"
        elif model_name_or_path == "CA-MTL-base":
            return "bert-base-cased"
        else:
            return model_name_or_path
