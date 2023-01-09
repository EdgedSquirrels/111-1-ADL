from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embedding_size = embeddings.size(1)
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_class = num_class
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(embeddings.size(1), hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_output_size, self.num_class),
            nn.Softmax()
        )
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * (2 if self.bidirectional else 1) * self.num_layers

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # id text intent
        x = batch['text']
        x = self.embed(x)
        x, h_n = self.rnn(x)
        h_n = torch.permute(h_n, (1, 0, 2))
        h_n = torch.reshape(h_n, (h_n.shape[0], -1))
        # print("h_n.shape:", h_n.shape)
        h_n = self.classifier(h_n)
        ret = {}
        ret['pred_logits'] = h_n
        ret['pred_labels'] = torch.argmax(h_n, axis=-1)
        return ret
        # raise NotImplementedError


class SeqTagger(SeqClassifier):
    @property
    def encoder_output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)
    
    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_output_size, self.num_class),
            nn.Sigmoid()
        )
        self.cnn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv1d(self.embedding_size, self.embedding_size, 11, padding=5),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # id tokens tags
        x = batch['tokens']
        x = self.embed(x)

        x = torch.permute(x, (0, 2, 1))
        x = self.cnn(x)
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.rnn(x)

        # print("SeqTagger forward shape:", x.shape)
        # x_dim: (batch, seq_len, hidden_size)


        x = self.classifier(x)
        # print("SeqTagger x.shape:", x.shape)
        ret = {}
        ret['pred_logits'] = x
        ret['pred_labels'] = torch.argmax(x, axis=-1)
        return ret
