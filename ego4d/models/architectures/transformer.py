import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype,  featuretype, nfeats, num_input_clips, num_verbs, num_nouns, num_intentions,num_actions_to_predict,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", semantic_dim=768, shared_embedding = True, pair_embedding=False, feature_dimension = 2304, **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.featuretype = featuretype
        self.shared_embedding = shared_embedding
        self.use_pair = pair_embedding
        self.latent_dim = latent_dim
        if not self.use_pair:
            self.latent_dim = latent_dim*2

        self.dim_features = feature_dimension ## Stablished by slowfast features dimensionality
        self.nfeats = nfeats
        self.num_input_clips = num_input_clips
        self.num_actions_to_predict = num_actions_to_predict

        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.num_intentions = num_intentions


        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        if self.featuretype == "onehot":
            ## we assume that input/forecast/intentions are one-hot, thus dimension (1) each
            if not self.shared_embedding:
                if self.use_pair:
                    self.pair_embedding = nn.Linear(self.latent_dim*2, self.latent_dim, bias=True)

                self.verbs_embeddings = nn.Embedding(num_embeddings=self.num_verbs, embedding_dim=int(self.latent_dim if self.use_pair else self.latent_dim/2))
                self.nouns_embeddings = nn.Embedding(num_embeddings=self.num_nouns, embedding_dim=int(self.latent_dim if self.use_pair else self.latent_dim/2))
                #nn.Parameter(torch.randn(self.num_verbs, self.latent_dim))
                #nn.Parameter(torch.randn(self.num_nouns, self.latent_dim))

        elif self.featuretype == "language":
            ## we assume that input/forecast/intention is sentence-embedded, thus dimension (732) each
            if self.use_pair:
                self.semantic_dim = semantic_dim
                self.pair_embedding = nn.Linear(self.semantic_dim*2, self.latent_dim, bias=True)

        else: #self.featuretype == 'vision'
            if not self.shared_embedding:

                self.vision_reducer = nn.Linear(self.dim_features, self.latent_dim, bias=True)
                self.verbs_embeddings = nn.Embedding(num_embeddings=self.num_verbs, embedding_dim=int(
                    self.latent_dim if self.use_pair else self.latent_dim / 2))
                self.nouns_embeddings = nn.Embedding(num_embeddings=self.num_nouns, embedding_dim=int(
                    self.latent_dim if self.use_pair else self.latent_dim / 2))

        self.muQuery_intention = nn.Parameter(torch.randn(self.num_intentions, self.latent_dim))
        self.sigmaQuery_intention = nn.Parameter(torch.randn(self.num_intentions, self.latent_dim))


        self.lnorm = nn.LayerNorm(self.latent_dim, eps=1e-05)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        observed_labels, intentions, forecast_labels = batch["observed_labels"], batch["intentions"], batch["forecast_embeds"]
        #mask = self.lengths_to_mask(bs)

        if self.featuretype == "onehot" or self.featuretype == "language": ## this or is new, just to check if pretrained works
            if not self.shared_embedding:
                observed_verbs = self.verbs_embeddings(observed_labels[:, :, 0])
                observed_nouns = self.nouns_embeddings(observed_labels[:, :, 1])

                inputs = torch.cat((observed_verbs, observed_nouns), dim=2)
                if self.use_pair:
                    inputs = self.pair_embedding(inputs)

                forecast_verbs = self.verbs_embeddings(forecast_labels[:, :, 0])
                forecast_nouns = self.nouns_embeddings(forecast_labels[:, :, 1])

                forecast = torch.cat((forecast_verbs, forecast_nouns), dim=2)
                if self.use_pair:
                    forecast = self.pair_embedding(forecast)
            else:
                inputs = observed_labels
                forecast = forecast_labels

        elif self.featuretype == "language":
            inputs = torch.cat([observed_labels[:, :, :, 0], observed_labels[:, :, :, 1]], dim=2)
            if self.use_pair:
                inputs = self.pair_embedding(inputs)

            forecast = torch.cat([forecast_labels[:, :, :, 0], forecast_labels[:, :, :, 1]], dim=2)
            if self.use_pair:
                forecast = self.pair_embedding(forecast)
        else: #vision
            if not self.shared_embedding:
                forecast_verbs = self.verbs_embeddings(forecast_labels[:, :, 0])
                forecast_nouns = self.nouns_embeddings(forecast_labels[:, :, 1])
                forecast = torch.cat((forecast_verbs, forecast_nouns), dim=2)
                inputs = self.vision_reducer(observed_labels)

            else:
                forecast = forecast_labels
                inputs = observed_labels

        x = torch.cat([inputs, forecast], dim=1).permute(1,0,2)

        # adding the mu and sigma queries
        xseq = torch.cat((self.muQuery_intention[intentions][None], self.sigmaQuery_intention[intentions][None], x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        xseq = self.lnorm(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        #muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        #maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq,
                                     # src_key_padding_mask=~maskseq,
                                     )

        return {"mu": final[0], "logvar": final[1], "forecast_embeds": forecast}

    def lengths_to_mask(self, bs):
        lengths = torch.tensor([self.num_actions_to_predict] * bs, dtype=int, device="cpu")
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, featuretype, nfeats,  num_input_clips, num_verbs, num_nouns, num_intentions,num_actions_to_predict,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, semantic_dim=768, shared_embedding = True, conditions_decoder="input_as_data",
                 pair_embedding=False, feature_dimension=2304, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.featuretype = featuretype
        self.shared_embedding = shared_embedding
        self.conditions_decoder = conditions_decoder
        self.use_pair = pair_embedding
        self.latent_dim = latent_dim
        if not self.use_pair:
            self.latent_dim = latent_dim*2

        self.dim_features = feature_dimension  ## Stablished by slowfast features dimensionality
        self.nfeats = nfeats
        self.num_input_clips = num_input_clips
        self.num_actions_to_predict = num_actions_to_predict

        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.num_intentions = num_intentions



        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        if self.featuretype == "onehot":
            if not self.shared_embedding:
                ## we assume that input/forecast/intentions are one-hot, thus dimension (1) each
                self.verbs_embeddings = nn.Embedding(num_embeddings=self.num_verbs, embedding_dim=int(self.latent_dim if self.use_pair else self.latent_dim/2))
                self.nouns_embeddings = nn.Embedding(num_embeddings=self.num_nouns, embedding_dim=int(self.latent_dim if self.use_pair else self.latent_dim/2))
                if self.use_pair:
                    self.pair_embedding = nn.Linear(self.latent_dim*2, self.latent_dim, bias=True)

        elif self.featuretype == "language":
            ## we assume that input/forecast/intention is sentence-embedded, thus dimension (732) each
            self.semantic_dim = semantic_dim
            if self.use_pair:
                self.pair_embedding = nn.Linear(self.semantic_dim*2, self.latent_dim, bias=True)
        else:  # vision
            if not self.shared_embedding:
                self.vision_reducer = nn.Linear(self.dim_features, self.latent_dim)
                ## we assume that input/forecast/intentions are one-hot, thus dimension (1) each
                self.verbs_embeddings = nn.Embedding(num_embeddings=self.num_verbs, embedding_dim=int(
                    self.latent_dim if self.use_pair else self.latent_dim / 2))
                self.nouns_embeddings = nn.Embedding(num_embeddings=self.num_nouns, embedding_dim=int(
                    self.latent_dim if self.use_pair else self.latent_dim / 2))

        self.actionBiases_intention = nn.Parameter(torch.randn(self.num_intentions, self.latent_dim))

        self.lnorm = nn.LayerNorm(self.latent_dim, eps=1e-05)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        

    def forward(self, batch):
        observed_labels, intentions, z = batch["observed_labels"], batch["intentions"], batch["z"]
        bs = observed_labels.shape[0]
        mask = self.lengths_to_mask(bs, z.device)

        # shift the latent noise vector to be the action noise
        z = z + self.actionBiases_intention[intentions]
        z = z[None]  # sequence of size 1

        if self.featuretype == "onehot" or self.featuretype == "language": ## this or is new, just to check if pretrained works
            if not self.shared_embedding:
                verbs = self.verbs_embeddings(observed_labels[:, :, 0])
                nouns = self.nouns_embeddings(observed_labels[:, :, 1])
                inputs = torch.cat((verbs, nouns), dim=2)
                if self.use_pair:
                    inputs = self.pair_embedding(inputs)
            else:
                inputs = observed_labels
        elif self.featuretype == "language":
            inputs = torch.cat((observed_labels[:, :, :, 0], observed_labels[:, :, :, 1]), dim=2)
            if self.use_pair:
                inputs = self.pair_embedding(inputs)
        else:
            inputs = observed_labels

        inputs = inputs.permute(1,0,2)

        timequeries = torch.zeros(self.num_actions_to_predict, bs, self.latent_dim, device=z.device)
        if self.conditions_decoder == "input_as_memory":
            conditions = torch.cat((z,inputs), axis=0)
        elif self.conditions_decoder == "input_as_data":
            conditions = z
            timequeries = torch.cat([inputs, timequeries], dim=0)
        else:
            print("Define correctly conditions_decoder argument in cfg.")
            conditions = torch.cat((z,inputs), axis=0)

        timequeries = self.sequence_pos_encoder(timequeries)
        timequeries = self.lnorm(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=conditions, tgt_key_padding_mask=~mask)
        # zero for padded area
        output[~mask.T] = 0

        batch["decoded_output"] = output[-self.num_actions_to_predict:, :,:]
        return batch

    def lengths_to_mask(self, bs, device):
        dim = self.num_actions_to_predict + self.num_input_clips if self.conditions_decoder =="input_as_data" else self.num_actions_to_predict
        lengths = torch.tensor([dim] * bs, dtype=int, device=device)
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask



class MultiHeadDecoder_cvae(nn.Module):
    def __init__(self, cfg, latent_dim=256, pair_embedding=False, heads_per_future=False, num_actions_to_predict=20,
                 separate_nouns_verbs = False, **kargs):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.use_pair = pair_embedding
        self.num_heads = num_actions_to_predict
        self.heads_per_future = heads_per_future
        head_classes =self.cfg.MODEL.NUM_CLASSES

        self.latent_dim = latent_dim * 2 if (
                    not separate_nouns_verbs and  not self.use_pair) else latent_dim  # how it was done before

        if not self.heads_per_future:
            self.heads = MultiTaskHead_cvae(
                dim_in=[self.latent_dim],
                num_classes=head_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
                separate_noun_verb=separate_nouns_verbs
            )
        else:
            self.heads = nn.ModuleList([MultiTaskHead_cvae(
                dim_in=[self.latent_dim],
                num_classes=head_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
                separate_noun_verb=separate_nouns_verbs
            ) for _ in range(num_actions_to_predict)])


    def forward(self, batch):
        x = batch["decoded_output"] # [Z, B, LatentDim]
        x = x.permute((1,0,2))# [B, Z, LatentDim]
        verbs = []
        nouns = []
        if self.heads_per_future:
            for (future_id, head) in enumerate(self.heads):
                v, n = head(x[:,future_id, :])
                verbs.append(v)
                nouns.append(n)
            batch["output"] = [torch.stack(verbs, axis=1), torch.stack(nouns, axis=1)]
        else:
            batch["output"] = self.heads(x)
        return batch


# For LTA models. One head per future action prediction
class MultiTaskHead_cvae(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
        separate_noun_verb=False
    ):
        super().__init__()
        self.test_noact = test_noact
        self.separate_noun_verb = separate_noun_verb
        self.dim_in = int(sum(dim_in)/2) if separate_noun_verb else sum(dim_in)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        #self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = []
        for n in num_classes:
            projs.append(nn.Linear(self.dim_in, n))
        self.projections = nn.ModuleList(projs)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError("{} is not supported as an activation" "function.".format(act_func))

    def forward(self, inputs):
        """

        :param inputs: dimensionality after the Transformer Decoder: [Z, B, Latent_Dim]
        :return: [[Z, B, Num_Classes_Verb], [Z, B, Num_Classes_Noun]]
        """
        feat = inputs
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        if self.separate_noun_verb:
            # Separates Verbs and Nouns Embeddings and design one head for each
            feat = torch.split(feat, self.dim_in, dim=-1)
            x = [projection(feat[ind]) for ind, projection in enumerate(self.projections)]
        else:
            # Performs fully convlutional inference.
            x = [projection(feat) for projection in self.projections]

        # Performs Activation Function (Softmax - Sigmoid)
        if not self.training:
            x = [self.act(x_i) for x_i in x]
        return x

# For LTA models. Embedding Look Up Table
class EmbeddingActions(nn.Module):
    def __init__(
        self,
        num_nouns,
        num_verbs,
        latent_dim,
        featuretype,
        pretrained_from = None,
        pair_embedding=False,
        feature_dimension = 2304,
            **kargs
    ):
        super().__init__()
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.featuretype= featuretype ## language or else. If language, pretraine embedding and adapt latent_dim to semantic_dim
        self.latent_dim = latent_dim
        self.use_pair = pair_embedding
        self.feature_dimension = feature_dimension

        if self.featuretype in "language":
            embedder = torch.load(pretrained_from)
            self.verbs_embeddings = nn.Embedding.from_pretrained(embedder["verbs"], freeze=False)
            self.nouns_embeddings = nn.Embedding.from_pretrained(embedder["nouns"], freeze=False)
            assert self.nouns_embeddings.embedding_dim == self.latent_dim
        else:
            self.verbs_embeddings = nn.Embedding(num_embeddings=self.num_verbs, embedding_dim=int(self.latent_dim))
            self.nouns_embeddings = nn.Embedding(num_embeddings=self.num_nouns, embedding_dim=int(self.latent_dim))
            if self.featuretype in  'vision':
                self.vision_reducer = nn.Linear(self.feature_dimension, int(self.latent_dim*2))

        if self.use_pair:
            self.pair_embedding = nn.Linear(self.latent_dim * 2, self.latent_dim, bias=True)

    def forward(self, batch):
        if self.featuretype not in 'vision':
            observed_labels = batch["observed_labels"]

            observed_verbs = self.verbs_embeddings(observed_labels[:, :, 0])
            observed_nouns = self.nouns_embeddings(observed_labels[:, :, 1])

            inputs = torch.cat((observed_verbs, observed_nouns), dim=2)
            if self.use_pair:
                inputs = self.pair_embedding(inputs)
            batch["observed_labels"] = inputs
        else:
            batch["observed_labels"] =  self.vision_reducer(batch["vision_features"])


        if "forecast_embeds" in batch:
            forecast_labels = batch["forecast_embeds"]
            forecast_verbs = self.verbs_embeddings(forecast_labels[:, :, 0])
            forecast_nouns = self.nouns_embeddings(forecast_labels[:, :, 1])

            forecast = torch.cat((forecast_verbs, forecast_nouns), dim=2)
            if self.use_pair:
                forecast = self.pair_embedding(forecast)
            batch["forecast_embeds"] = forecast
        return batch


