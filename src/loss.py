import torch

class WeightedCrossEntropyLoss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        
    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss