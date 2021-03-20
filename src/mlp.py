import gzip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


from .hyperparameters import (
    DROPOUT,
    EMBEDDINGS_SIZE,
    EPOCHS,
    FREEZE_EMBEDINGS,
    HIDDEN_LAYERS,
)


class MLPClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=HIDDEN_LAYERS,
                 dropout=DROPOUT,
                 vector_size=EMBEDDINGS_SIZE,
                 freeze_embedings=FREEZE_EMBEDINGS):
        super().__init__()

        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)

        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)

        with gzip.open(pretrained_embeddings_path, "rt") as fh:
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] = (
                        torch.FloatTensor([float(n) for n in vector.split()])
                    )

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings_matrix,
            freeze=freeze_embedings,
            padding_idx=0,
        )
        self.hidden_layers = [
            nn.Linear(vector_size, hidden_layers[0])
        ]

        hidden_layers_iterator = zip(hidden_layers[:-1], hidden_layers[1:])
        for input_size, output_size in hidden_layers_iterator:
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )

        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)

        x = self.output(x)
        return x


