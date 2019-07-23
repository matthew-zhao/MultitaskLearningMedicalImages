import torch

class WeightedCrossEntropyLoss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        
    def forward(self, inputs, targets, phase):
        print(self.Wt1[phase].size(), self.Wt0[phase].size())
        print(targets.size())
        print(inputs.size())
        loss = - (self.Wt1[phase] * (targets * torch.log(inputs)) + self.Wt0[phase] * ((1 - targets) * torch.log(1 - inputs)))
        return loss