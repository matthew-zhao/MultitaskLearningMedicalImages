import torch.nn as nn
from torch.nn.functional import softmax, relu, avg_pool2d

class _Encoder(nn.Module):
    def __init__(self, pretrained_model, input_size):
        super(_Encoder, self).__init__()
        self.pretrained_model = pretrained_model
        self.input_size = input_size

    def forward(self, x):
        x = self.pretrained_model.features(x)
        out = relu(x, inplace=True)
        out = avg_pool2d(out, kernel_size=int(self.input_size / 32), stride=1).view(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        return out


class _Decoder(nn.Module):
    def __init__(self, output_size):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            #nn.Linear(128*8*8, 1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class _Model(nn.Module):
    def __init__(self, output_size, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(output_size=output_size)

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