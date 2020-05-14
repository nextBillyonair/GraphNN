import torch
from torch.nn import BCEWithLogitsLoss, Module

#Loss for LP
class MaskedLoss(Module):

    def __init__(self):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(reduction='none')

    def forward(self, output, node_set, adj_matrix):
        # only look at train nodes
        adj_matrix = [:, list(node_set.keys()), :]
        output = [:, list(node_set.keys()), :]

        num_links = adj_matrix.sum() # num pos links not hidden

        # neg sample
        flat_adj_matrix = adj_matrix.view(-1)
        idx = torch.randperm(flat_adj_matrix.size(0))[:int(num_links)]
        mask = torch.zeros_like(adj_matrix).view(-1)
        mask[idx] = 1.
        mask = mask.reshape(*adj_matrix.size())

        # combine with adj_matrix
        mask = (mask.bool() | adj_matrix.bool()).float()

        # take loss
        loss = self.criterion(output[mask == 1.].view(-1), adj_matrix[mask == 1.].view(-1))

        # reduce
        loss = loss.sum() / (2 * num_links)
        return loss
