import torch
import torch.nn as nn



class CAE(nn.Module):
    def __init__(self, encoder, decoder, multihead, embedder, mlpmixer, shared_embedding, device, lambdas, latent_dim, featuretype, num_actions_to_predict, pair_embedding, **kwargs):
        super().__init__()

        self.embedder = embedder
        self.mlpmixer = mlpmixer
        self.encoder = encoder
        self.decoder = decoder
        self.multihead = multihead

        self.featuretype = featuretype
        self.shared_embedding = shared_embedding


        self.lambdas = lambdas

        self.use_pair = pair_embedding
        self.latent_dim = latent_dim
        if not self.use_pair:
            self.latent_dim = latent_dim*2

        self.device = device

        
        self.losses = list(self.lambdas) + ["mixed"]
        self.num_actions_to_predict = num_actions_to_predict


    
    def forward(self, batch):
        if self.params.featuretype in 'vision':
            batch.update(self.resume_visual(batch))
        elif self.shared_embedding:
            batch.update(self.embedder(batch))
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))
        batch.update(self.multihead(batch))
        return batch

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]
        
        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]
            
    def generate(self, intentions, observed_labels, k=5,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        if k is None:
            k = 5
        n_intentions = len(intentions)
            
        y = intentions.to(self.device).repeat(k)  # (view(nspa, nats))
        if len(observed_labels.shape) == 3:
            x = observed_labels.to(self.device).repeat(k,1,1)
        else:
            x = observed_labels.to(self.device).repeat(k,1,1,1)

        if noise_same_action == "random":
            if noise_diff_action == "random":
                z = torch.randn(k*n_intentions, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_same_action = torch.randn(k, self.latent_dim, device=self.device)
                z = z_same_action.repeat_interleave(n_intentions, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(n_intentions, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(n_intentions, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, k, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(k*n_intentions, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(n_intentions, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(n_intentions, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((k, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")

        k_lab = 'observed_labels' if self.featuretype not in 'vision' else 'vision_features'
        batch = {"z": fact*z, "intentions": y, k_lab : x}
        if self.featuretype == 'vision':
            batch.update(self.resume_visual(batch))  # [B x N x D] = [B x 2 x 2304]
        if self.shared_embedding:
            batch.update(self.embedder(batch)) # [B x N x L] = [B x 2 x latent_dim]

        batch.update(self.decoder(batch))
        batch.update(self.multihead(batch))
        return self.getResults(batch["output"], k)

    def getResults(self, x, k):
        results = []
        for head_x in x: # [verbs, nouns]
            total, z, num_classes = head_x.shape
            bs = int(total / k)
            preds = head_x.argmax(2)
            results.append(torch.stack(torch.split(preds, bs, dim=0), dim=1))
        return results
    
    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]
