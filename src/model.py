import torch.nn as nn
from torch.nn.functional import softmax, relu, avg_pool2d, sigmoid

class _Encoder(nn.Module):
    def __init__(self, pretrained_model, input_size):
        super(_Encoder, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.features.norm5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.input_size = input_size
        self.dim_feats = self.pretrained_model.last_linear.in_features
        print(self.dim_feats)

    def forward(self, x):
        x = self.pretrained_model.features(x)
        #out = relu(x, inplace=True)
        #out = avg_pool2d(out, kernel_size=int(self.input_size / 32), stride=1).view(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        return x


class _Decoder(nn.Module):
    def __init__(self, in_features, output_size):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            #nn.Linear(128*8*8, 1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class _Model(nn.Module):
    def __init__(self, output_size, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(in_features=encoder.dim_feats, output_size=output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def Model(num_tasks, pretrained_model, input_size):
    encoder = _Encoder(pretrained_model=pretrained_model, input_size=input_size)
    # multitask case
    #if isinstance(num_tasks, list):
    return [_Model(output_size=class_size, encoder=encoder) for class_size in num_tasks]

    # single task case
    #return _Model(output_size=num_tasks, encoder=encoder)