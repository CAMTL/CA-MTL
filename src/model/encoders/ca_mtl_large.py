import math
import torch
import torch.nn as nn
from transformers.modeling_bert import (
    BertEmbeddings,
    BertPooler,
    BertPreTrainedModel,
    BertAttention,
    BertIntermediate,
    BertLayer,
    BertSelfOutput,
)

from src.model.encoders.conditional_modules import FiLM, CBDA, ConditionalLayerNorm, ConditionalBottleNeck


class MyBertSelfAttention10(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.max_seq_length = config.max_seq_length
        assert config.hidden_size % self.max_seq_length == 0, \
            "Block decomposed attention will only work if this condition is met."
        self.num_blocks = config.hidden_size // self.max_seq_length
        self.cond_block_diag_attn = CBDA(
            config.hidden_size, math.ceil(self.max_seq_length / self.num_blocks), self.num_blocks
        )  # d x L/N

        self.random_weight_matrix = nn.Parameter(
            torch.zeros(
                [config.max_seq_length, math.ceil(self.max_seq_length / self.num_blocks)]
            ),
            requires_grad=True,
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task_embedding=None,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_value_layer = self.value(hidden_states)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        mixed_key_layer = self.key(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)

        attention_scores2 = self.cond_block_diag_attn(
            x_cond=task_embedding,
            x_to_film=self.random_weight_matrix,
        )

        attention_scores = attention_scores1 + attention_scores2.unsqueeze(1)

        # b x seq len x hid dim

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # y = ax + b(task_emb)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores
        )  # b x num heads x seq length x head dim

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class MyBertSelfOutput10(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = ConditionalLayerNorm(config.hidden_size, config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, task_embedding, task_id):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, task_embedding, task_id)
        return hidden_states


class MyBertOutput10(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = ConditionalLayerNorm(config.hidden_size, config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, task_embedding, task_id):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, task_embedding, task_id)
        return hidden_states


class MyBertAttention10(BertAttention):
    def __init__(self, config, add_conditional_layernorm=True):
        super().__init__(config)
        self.self = MyBertSelfAttention10(config)
        self.add_conditional_layernorm = add_conditional_layernorm
        if add_conditional_layernorm:
            self.output = MyBertSelfOutput10(config)
        else:
            self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task_embedding=None,
        task_id=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            task_embedding=task_embedding,
        )
        if self.add_conditional_layernorm:
            attention_output = self.output(self_outputs[0], hidden_states, task_embedding, task_id)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class MyBertLayer10(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MyBertAttention10(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = MyBertOutput10(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task_embedding=None,
        task_id=None
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, task_embedding=task_embedding, task_id=task_id
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, task_embedding, task_id)
        outputs = (layer_output,) + outputs
        return outputs


class BertLayer10(BertLayer):
    """Same as BertLayer but with different inputs"""
    def __init__(self, config):
        super().__init__(config)
        self.attention = MyBertAttention10(config, add_conditional_layernorm=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task_embedding=None,
        task_id=None
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, task_embedding=task_embedding, task_id=task_id
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class MyBertEncoder10(nn.Module):
    def __init__(self, config, tasks):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.task_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        num_bert_layers = config.num_hidden_layers//2 + config.num_hidden_layers % 2
        num_mybert_layers = config.num_hidden_layers//2
        assert num_bert_layers+num_mybert_layers == config.num_hidden_layers
        self.layer = nn.ModuleList(
            [BertLayer10(config) for _ in range(num_bert_layers)] +
            [MyBertLayer10(config) for _ in range(num_mybert_layers)]
        )
        assert len(self.layer) == config.num_hidden_layers
        #FiLM6
        self.adapter_layer = nn.ModuleList(
            [
                ConditionalBottleNeck(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task_type=None,
        task_embedding=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        task_embedding = self.task_transformation(task_embedding)
        hidden_film = torch.zeros_like(hidden_states)
        for i, (layer_module, adapter_module) in enumerate(zip(self.layer, self.adapter_layer)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                task_embedding,
                task_type
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            # FiLM layer with a skip connection
            hidden_film = adapter_module(
                x_cond=task_embedding, hidden_states=hidden_states + hidden_film
            )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_film,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class CaMtlLargeEncoder(BertPreTrainedModel):
    r"""
    # NOTE: Combination of: (might work best for large) and uses:
        -- block diagonal attention
        -- conditional layer norm for the top half layers base=6-10 and large=11-22, top layer excluded
        -- conditional bias attention term to the original attention matrix
        -- down/projection bottleneck for all layers
        -- conditional alignment after the embedding layer only
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertFiLMModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config, data_args=None, **kwargs):
        super().__init__(config)
        tasks = data_args.tasks
        self.task_id_2_task_idx = {i: i for i, t in enumerate(tasks)}
        self.config = config
        self.config.num_tasks = len(tasks)
        config.max_seq_length = data_args.max_seq_length
        self.task_type_embeddings = nn.Embedding(len(tasks), config.hidden_size)
        self.conditional_alignment = FiLM(
            config.hidden_size, config.hidden_size
        )  # FiLM5

        self.embeddings = BertEmbeddings(config)
        self.encoder = MyBertEncoder10(config, tasks)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        """Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in:
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        task_type = self._create_task_type(kwargs["task_id"])
        task_embedding = self.task_type_embeddings(task_type)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.to(
                    torch.long
                )  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        embedding_output = self.conditional_alignment(
            x_cond=task_embedding,
            x_to_film=embedding_output,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            task_type=task_type,
            task_embedding=task_embedding,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def _create_task_type(self, task_id):
        task_type = task_id.clone()
        unique_task_ids = torch.unique(task_type)
        unique_task_ids_list = (
            unique_task_ids.cpu().numpy()
            if unique_task_ids.is_cuda
            else unique_task_ids.numpy()
        )
        for unique_task_id in unique_task_ids_list:
            task_type[task_type == unique_task_id] = self.task_id_2_task_idx[
                unique_task_id
            ]
        return task_type
