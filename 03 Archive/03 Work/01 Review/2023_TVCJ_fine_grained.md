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
2. The authors should verify the effectiveness of the LSTM module by adding a group ablation study to comparing classification results with and without LSTM.


### Others questions
1. The author mentions “we optimize a single network with weak supervision” in the last sentence of Section 2.1. But according to the method described in Section 3, supervised learning is adopted in this paper. Please explain how does it use weak supervision ?
2. According to Section 3.3, the TransFGVC consists of Swim-Transformer、LSTM and a classifier. But the author does not introduce the detail of the classifier.
3. The author uses “accuracy” and “precision” to show the performance of TransFGVC in Conclusion section. But these two words have different meanings in English, please check them! 
4. There are some other issues. For example, the caption of Table 6 does not match the table and the table is far away from the introduction of it. And what is the meaning of the “B” in Table 6?  



### Adequate reference to previous work
The missing references are as follows:
[1] A novel plug-in module for fine-grained visual classification
[2] MetaFormer: A Unified Meta Framework for Fine-Grained Recognition

The above papers have not been published.

