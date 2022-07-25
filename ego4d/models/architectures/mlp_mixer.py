import torch
from torch import nn
from functools import partial
import numpy as np
pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MLPMixer(nn.Module):
    def __init__(self, num_features, feature_dimension,
                 depth, num_classes, expansion_factor=4, expansion_factor_token=0.5, dropout=0., reduce_to_class=True,
                 action_loss=False, position_encoder=False, test_noact=False):
        super().__init__()

        num_patches = num_features
        self.feature_dimension = feature_dimension
        self.depth = depth
        self.num_classes = num_classes
        self.action_loss = action_loss
        self.reduce_to_class = reduce_to_class
        self.test_noact = test_noact

        self.pos_encoder = position_encoder
        if self.pos_encoder:
            self.positional_encoder = PositionalEncoding(feature_dimension, dropout, max_len=num_patches)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.mixer = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(feature_dimension, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(feature_dimension,
                                FeedForward(feature_dimension, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(feature_dimension)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, feature_dimension))

        if self.reduce_to_class:
            self.double_head = type(num_classes) == list
            if self.double_head:
                self.head = nn.ModuleList([nn.Linear(feature_dimension, num) for num in num_classes]) # [verb, noun]
            else:
                self.head = nn.Linear(feature_dimension, num_classes)
            self.act = nn.Softmax(dim=1)

    def forward(self, x):
        """

        :param x: Input parameter
        :return: Depends on the given configuration
            reduce_to_class: determines whether the output has to be a Classification or a Feature (determines the use of Linear Heads)
                reduce_to_class is used when there is a need of classifying, such as to determine intention or when classifying action pair verb-noun

            double head: this attribute is used when the NUM_CLASSES passed as argument is a list, then creating a linear layer to determine the action class for each step.
            training: determines whether to apply softmax function or not (cross_entropy_loss already applies softmax)
            action_loss: determines if only return the classified class or also return the features used for teh classification (used afterwards for intention classification)
        """
        if self.pos_encoder: ## to input the pos_encoder, we need dimensionality [NUM_PATCHES, B, Latent_dim]
            x = x.permute(1,0,2)
            x = self.positional_encoder(x)
            x = x.permute(1,0,2)

        x = self.mixer(x)
        x = self.avg_pool(x).squeeze()
        if len(x.shape) == 1: # Case when Batch Size=1
            x = x.unsqueeze(dim=0)
        if self.reduce_to_class:
            if self.double_head:
                y = [self.head[i](x) for i in range(len(self.num_classes))]# [verb, noun]
                if not self.training and not self.test_noact:
                    y =  [self.act(i) for i in y]
                if self.action_loss:
                    return y, x
                else:
                    return y
            if not self.training:
                return self.act(self.head(x))
            return self.head(x)
        return x
