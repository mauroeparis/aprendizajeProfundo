import gzip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperparameters import (
    EMBEDDINGS_SIZE,
    FC_OUTPUT,
    FILTERS_COUNT,
    FILTERS_LENGTH,
    FREEZE_EMBEDINGS,
)


class CNNClassifier(nn.Module):

    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
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

        self.convs = []
        for filter_lenght in FILTERS_LENGTH:
            self.convs.append(
                nn.Conv1d(vector_size, FILTERS_COUNT, filter_lenght)
            )
        self.convs = nn.ModuleList(self.convs)

        self.fc = nn.Linear(FILTERS_COUNT * len(FILTERS_LENGTH), FC_OUTPUT)
        self.output = nn.Linear(FC_OUTPUT, n_labels)
        self.vector_size = vector_size

    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])

    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))
        x = torch.sigmoid(self.output(x))
        return x
