import torch.nn as nn

from base.layers import Conv2dWithConstraint, LinearWithConstraint
from utils.utils import init_weight


class OneDCNN(nn.Module):
    def __init__(self,
                 n_classes,
                 input_shape,
                 F1=None,
                 D=None,
                 F2=None,
                 T1=None,
                 T2=None,
                 P1=None,
                 P2=None,
                 stride=None,
                 drop_out=None,
                 pool_mode=None,
                 init_weight_method=None,
                 *args,
                 **kwargs):
        super().__init__()
        # b, c, s, t = input_shape
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D

        # Conv 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(22, 20, 10, stride=1),
            nn.MaxPool1d(2, stride=2)
        )

        # Conv 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 40, 5, stride=1),
            nn.MaxPool1d(2, stride=2)
        )

        # Conv 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(40, 80, 3, stride=1),
            nn.MaxPool1d(2, stride=2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # LinearWithConstraint(F2 * 3, n_classes, max_norm=0.25)
            LinearWithConstraint(80 * 59, n_classes, max_norm=0.25)
        )

        init_weight(self, init_weight_method)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out
