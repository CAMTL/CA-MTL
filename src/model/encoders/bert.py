from transformers import BertModel


class _BertEncoder(BertModel):  # Adding a prefix since BertEncoder exists in huggingface lib
    def __init__(self, config, **kwargs):
        super(_BertEncoder, self).__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return super(_BertEncoder, self).forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

