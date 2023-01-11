import torch.nn as nn

class ActionHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        super(ActionHead, self).__init__()

        self.test_noact = test_noact
        self.avg_pool = nn.AdaptiveAvgPool2d((1, sum(dim_in)))

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = []
        for n in num_classes:
            projs.append(nn.Linear(sum(dim_in), n, bias=True))
        self.projections = nn.ModuleList(projs)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        # Perform dropout.
        x = self.avg_pool(x).squeeze(1)
        feat = x
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        x = []
        for projection in self.projections:
            x.append(projection(feat))

        # Performs fully convlutional inference.
        if not self.training:
            if not self.test_noact:
                x = [self.act(x_i) for x_i in x]
        return x
