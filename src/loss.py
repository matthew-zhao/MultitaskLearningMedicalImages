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
        loss = - (self.Wt1[phase] * (targets * log_softmax(inputs, 1)) + self.Wt0[phase] * ((1 - targets) * torch.log(1 - inputs)))
        return loss


class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    '''
    0) Mask ones that are equal to value_to_ignore
    1) Sum over the masked observations
    2) Average/mean over datapoints (images) in minibatch
       https://stats.stackexchange.com/questions/306862/cross-entropy-versus-mean-of-cross-entropy
    '''
    def __init__(self, weight=None, pos_weight=None, value_to_ignore=-1.0):
        super(MaskedBCEWithLogitsLoss, self).__init__(weight, None, None, 'none', pos_weight)
        self.value_to_ignore = value_to_ignore

    def forward(self, input, target):
        unreduced_output = super().forward(input, target)
        replacement_zeros = torch.zeros(unreduced_output.size(0), unreduced_output.size(1))
        masked_unreduced_outputs = torch.where(target == value_to_ignore, replacement_zeros, unreduced_output)
        summed_output_over_observations = torch.sum(masked_unreduced_output, dim=1)
        avg_output_over_data = torch.mean(summed_output_over_observations)
        return avg_output_over_data