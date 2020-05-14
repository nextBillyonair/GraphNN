# GraphNN
Test Repo for Graph Transformer Models

## TODO
Retest all modules so they actually work, most should at the moment

## Purpose
This repo represents a set of torch modules used to manipulate graph information.
Usually, we have a graph `G`, with a set of vertices `V`. Each vertex has a
feature vector `E`, representing the embedding size. For example, using a 300 TF-IDF or
BOW representation on a graph with 1440 vertices results in a torch tensor of
size `(G, V, E) -> (1, 1440, 300)`.

The idea here being, with an adjacency matrix `G, V, V`, where it is a
binary tensor such that Row -> Col. With this, we can easily mask nodes in the
multihead attention.

A sample model would be:

`Nodes -> Linear Projection -> N Graph Attention Layers -> PairwiseBilinear`

for a link prediction task, and use negative sampling to train.

For classification, an output dense layer should suffice,


## Modules
* `Graph Attention Layer`
  * `GraphAttentionNetwork`
  * `GraphAttentionLayer`
* `Node Transformer`
  * `PositionalEncoding`
  * `NodeTransformer`
  * `TransformerNetwork`
  - These are unstable due to the large amount of memory used (1, V, S, E)
* `Transformer`
  * `Transformer`
  * `TransformerDecoder`
    - Still need to figure out use case for Networks
* `Multihead Attention`
  * `MultiheadAttention`
* `Residual`
  * `AdditiveResidual`
  * `GeneralResidual`  
  * `DotResidual`  
* `Pairwise`
  * `PairwiseBilinear`
  * `PairwiseDot`  
  * `PairwiseDistance`  
