import torch
import torch.nn as nn
from torch.nn import Module, Linear

class PairwisePrediction(Module):
    def __init__(self, d_model, dropout = 0.2, leaky_relu_negative_slope=0.2, share_weights=True):
        super().__init__()
        self.linear_left = Linear(d_model, d_model)
        if share_weights:
            self.linear_right = self.linear_left
        else:
            self.linear_right = Linear(d_model, d_model)
            
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2*d_model, 1) # to logits
        
    def forward(self, embeddings):
        #[N, S, E]
        bz, sz, ez = embeddings.shape
        left_embed = self.dropout(self.linear_left(embeddings)) #N, S, E
        right_embed = self.dropout(self.linear_right(embeddings)) #N, S, E
        left_embed = left_embed.repeat(1, sz, 1) #[N, S*S, E] -> regular so a_o, a_1, a_2....a_o, a_1, a_n
        right_embed = right_embed.repeat_interleave(sz, dim=1) #[N, S*S, E] -> interleaved so a_o, a_o, a_o...a_n,a_n
        pairwise_embed = torch.cat((left_embed, right_embed), dim=-1).reshape(bz, sz, sz, -1) #[N, S, S, 2*E] can always sum to save parameters
        link_logits = self.classifier(self.activation(pairwise_embed)) #N, S, S, 1
        return link_logits #N, S, S, 1

    
class PairwiseBilinear(Module):
    def __init__(self, input_size=64, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size
        self.linear = Linear(input_size, input_size, bias=bias)

    def forward(self, embeddings):
        dot_prod = torch.bmm(self.linear(embeddings), embeddings.transpose(-1, -2))
        return dot_prod


class PairwiseDot(Module):
    def forward(self, embeddings):
        dot_prod = torch.bmm(embeddings, embeddings.transpose(-1, -2))
        return dot_prod


class PairwiseDistance(Module):
    def forward(self, embeddings, p=2):
        n, d = embeddings.size(-2), embeddings.size(-1)
        x = embeddings.expand(n, n, d)
        y = embeddings.expand(n, n, d).transpose(-2, -3)
        dist = torch.pow(x - y, p).sum(-1).unsqueeze(0)
        return dist
