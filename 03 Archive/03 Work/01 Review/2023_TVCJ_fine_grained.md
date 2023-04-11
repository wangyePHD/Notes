## TransFGVC: Transformer-based Fine-Grained Visual Classification

### Paper sunmmary
This paper proposed a method called TransFGVC that consists of Swin Transformer and a LSTM network to achieve fine-grained visual classification task. The Swin Transformer is used to extract the local visual features, and then LSTM model is utilized to learn the global information based on these local features. Furthermore, this paper developed Birds-267-2022, a fine-grained dataset which has 267 categories and 12233 images.


### Imapct and originality of contribution
1. This paper proposed a reasonable method but the novelty is not enough. The approach just combine Swin Transformer and LSTM model simply,  it does not design a new module or a new learning paradigms for learning local and global features.
2. The comparisons with SOTA methods are insufficient. For example, in CUB-200-2011  and NABirds settings, this paper does not compare TransFGVC to the existing SOTA method, such as PIM[1] and MetaFormer[2].

### Organization and clarity of presentation
1. All images in this paper have some ambiguity. 
2. The caption of Table 6 seems incorrect.

### Technical correctness and quality of results
1. The technical correctness is fine but this paper lacks of comparisons with the existing SOTA methods.

### Adequate reference to previous work
The missing references are as follows:
[1] A novel plug-in module for fine-grained visual classification
[2] MetaFormer: A Unified Meta Framework for Fine-Grained Recognition

The above papers have not been published.