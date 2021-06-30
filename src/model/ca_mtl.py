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
    freeze_encoder_layers: str = field(
        default=None,
        metadata={"help": "Freeze encoder layers. format: <start_layer>-<end_layer>"},
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

    def freeze_encoder_layers(
        self,
        model_args,
        unfrozen_modules=[
            "random_weight_matrix",
            "film.gb_weights",
            "ln_weight_modulation.gb_weights",
            "adapter",
        ],
    ):
        if model_args.freeze_encoder_layers is not None:
            start_layer, end_layer = model_args.freeze_encoder_layers.split("-")

            for name, param in self.bert.named_parameters():
                requires_grad = True
                match = re.match(self.bert.get_layer_regexp(), name)
                if match:
                    layer_number = int(match.groups()[0])
                    requires_grad = not int(start_layer) <= layer_number <= int(
                        end_layer
                    ) or any([module in match.string for module in unfrozen_modules])
                elif name.startswith("embedding"):
                    requires_grad = False
                param.requires_grad = requires_grad

        for name, param in self.bert.named_parameters():
            logger.info(
                "%s - %s", name, ("Unfrozen" if param.requires_grad else "FROZEN")
            )
