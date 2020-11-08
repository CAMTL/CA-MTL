import torch
import numpy
from scipy.stats import entropy
from transformers import glue_tasks_num_labels
from torch.nn import MSELoss, CrossEntropyLoss, Softmax, Dropout, Linear, Softmax


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, task_name):
        super().__init__()
        self.num_labels = glue_tasks_num_labels[task_name]
        self.dropout = Dropout(0.1)
        self.model = Linear(hidden_size, self.num_labels)

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        loss = None
        pooled_output = self.dropout(pooled_output)
        logits = self.model(pooled_output)

        batch_entropy = self.calculate_entropy(logits)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.long().view(-1))

        return logits, loss, batch_entropy

    def calculate_entropy(self, logits):
        probas = Softmax(dim=1)(logits.detach())
        samples_entropy = entropy(probas.transpose(0, 1).cpu())
        even_preds = numpy.array(
            [[1 / self.num_labels for _ in range(self.num_labels)]]
        )
        max_entropy = entropy(even_preds.T)
        epsilon = 1e-5
        samples_entropy = samples_entropy / (max_entropy.item() + epsilon)
        return torch.tensor(samples_entropy)
