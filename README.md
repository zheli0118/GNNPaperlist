# 基于图神经网络的多模态融合 papers
&nbsp;&nbsp; 个人于23年9月至11月对基于图神经网络的多模态融合的论文推荐和总结



## 图神经网络 papers

### 图神经网络综述
1. **Self-Supervised Learning of Graph Neural
Networks: A Unified Review**. *Shuiwang Ji（ 23年的fellow，一年5篇 TPAMI 的狠人）*. ***TPAMI 2022*** [[pdf](https://ieeexplore.ieee.org/document/9764632?denied=)] 

    &nbsp;&nbsp;至今还都不太懂，常读常新吧，这篇比较难

2. **A Comprehensive Survey on Graph Neural Networks**. *Philip S. Yu（Life Fellow）*. ***TNNLS 2021*** [[pdf](https://ieeexplore.ieee.org/document/9046288)] 

    &nbsp;&nbsp;对新手比较友好，我的前期就是靠这个入门的

### 同构图基础（谱域和空间域）

1. **Semi-Supervised Classification with Graph Convolutional Networks**. *Thomas N. Kipf( 牛逼，博二发这篇，物理专业出身 ), Max Welling（大老板，VAE创始人），都来自马普所*. ***ICLR 2017*** [[pdf](https://browse.arxiv.org/pdf/1609.02907.pdf)] [[original code](https://github.com/tkipf/gcn)] [[pyg code](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html)]&nbsp;&nbsp;

    &nbsp;&nbsp;本文是第三代谱图神经网络的论文，数学推论很有亮点。注意在原文中训练图的方式是 transductive 的， 而且不能生成 unseen 点的表征；GCN需要全图计算，计算量大，大图爆显存

2. **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**. *Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst.* ***NIPS 2016*** [[pdf](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf)] 

    &nbsp;&nbsp;第二代谱图神经网络，使用切比雪夫展开，也就是所谓的 chebnet，第三代本质就是把 chebnet 做进一步的简化

3. **Inductive Representation Learning on Large Graphs**. *Jure Leskovec （ 斯坦福神仙，GIN、node2vec 创始人，DGL库开发者，OGB 数据集开源者，年轻又斯文 ），William L Hamilton, Junlin Yang.* ***NIPS 2017*** [[pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)] [[original code](https://github.com/williamleif/GraphSAGE)] [[pyg code](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html)]

    &nbsp;&nbsp;本文是 GraphSAGE 的原作，空间域图。我认为是 GCN 和 GAT 承上启下的工作（尽管他们几乎是同一时期），使GCN扩展成归纳学习任务，对未知节点起到泛化作用，也就是所谓的 inductive；训练 graphSAGE 的时候甚至可以不用全图信息，解决了 GAT 和 GCN 的痛点；本文也首次提出 GCN 是一种特殊的空间域图。本文还考虑了多种不同的聚合方式，会让你有一个新的看图的视野。<u>一般认为本文比较适合图在工业界的落地应用。</u>

4. **Graph Attention Networks**. *Yoshua Bengio（这位不用多说了吧）*. ***ICLR 2018*** [[pdf](https://browse.arxiv.org/pdf/1710.10903.pdf)] [[original code]( https://github.com/PetarV-/GAT.)] [[pyg code](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html)] 

    &nbsp;&nbsp;GCN 不同节点的边视为一样的，GAT 参考 Transformer 的思想，基于点之间的注意力分配边的权重。 GAT 尤其适合有向图，因为不同方向得到的注意力是不一样的。注意，这里的边的权重只考虑了 src 节点与 dst 节点的关系，忽视了 src 节点与 dst 的邻居节点之间的关系，后续有 GraphFormer、GOAT 等来完善。

5. **HOW POWERFUL ARE GRAPH NEURAL NETWORKS?**.  *Jure Leskovec（斯坦福神仙）*. ***ICLR 2019*** [[pdf](https://browse.arxiv.org/pdf/1810.00826.pdf)] 

    &nbsp;&nbsp;关注图表征任务，很有帮助

### 同构图进阶
1. **non-local graph neural networks**.  *Shuiwang Ji*. ***TPAMI 2022*** [[pdf](https://ieeexplore.ieee.org/document/9645300)][[code](https://github.com/divelab/Non-Local-GNN)] 

    &nbsp;&nbsp;关注 disassortative graphs 

2. **An End-to-End Deep Learning
Architecture for Graph Classification**.  *Shuiwang Ji*. ***AAAI 2018*** [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11782)][[code](https://github.com/muhanzhang/DGCNN)] 

    &nbsp;&nbsp;DGCNN。简单看下思想，这篇我代码没跑。这篇提出了根据一定算法排序点，这样就不用考虑点的排列不变性(Permutation Invariance)，本来针对这个性质只能用 sum、mean 来表征全图，现在可以不考虑这个性质，用 LSTM 等来表征（不一定是这篇文章的，但是这篇文章开了个头）

3. **Graph Ordering Attention Networks**.  *Michail Chatzianastasis，Michalis Vazirgiannis*. ***AAAI 2023*** [[pdf](https://browse.arxiv.org/pdf/2204.05351.pdf)][[code](https://github.com/MichailChatzianastasis/GOAT)] 

    &nbsp;&nbsp;GAT 加上上述 DGCNN 的想法进行图表征

4. **Representation Learning on Graphs with Jumping Knowledge Networks**.  *Keyulu Xu*. ***ICLR 2018*** [[pdf](https://browse.arxiv.org/pdf/1806.03536.pdf)]

    &nbsp;&nbsp; JK-NET。只要是图就有过平滑问题，而且边缘节点有可能在信息传递的过程中只能拿到一小片区域的节点信息。作者将多层图神经网络（各种）的各层表征  concat 后送进 LSTM，本质是根据任务选出最适合的那层，从而选出最适合的邻居节点，绕开了过平滑问题。有意思。

5. **Do Transformers Really Perform Bad for Graph Representation**.  *Chengxuan Ying*. ***NIPS 2021*** [[pdf](https://browse.arxiv.org/pdf/2106.05234.pdf)]

    &nbsp;&nbsp; Graphormer。之前使用注意力机制解决图问题的模型都是在点级别的任务，在全图任务上表现不佳。作者本质上是在transformer上面针对图加了各种编码，包括度编码、空间关系编码、边信息编码。


### 异构图基础（谱域和空间域）

1. **Modeling Relational Data with Graph Convolutional**.  *Thomas N. Kipf*. ***2017*** [[pdf](https://browse.arxiv.org/pdf/1703.06103.pdf)]

    &nbsp;&nbsp;大名鼎鼎的RGCN。是 GCN 在异构图上的应用



2. **Heterogeneous Graph Attention Network**.  *Xiao Wang, Houye Ji*. ***WWW 2019*** [[pdf](https://browse.arxiv.org/pdf/1903.07293.pdf)][[code](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model.py)]

    &nbsp;&nbsp;大名鼎鼎的HAN。是 GAT 在异构图上的应用。

3. **Heterogeneous Graph Transformer**.  *Ziniu Hu*. ***WWW 2020*** [[pdf](https://browse.arxiv.org/pdf/2003.01332.pdf)][[code](https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py)]

    &nbsp;&nbsp;大名鼎鼎的HGT。他对注意力机制的改造很有意思.

4. **muxGNN: Multiplex Graph Neural Network for Heterogeneous Graphs**.  *Joshua Melton and Siddharth Krishnan*. ***TPAMI 2023*** [[pdf](https://ieeexplore.ieee.org/document/10086644)]

    &nbsp;&nbsp;最新的对异构图的分析，但是我还看不懂。后续一定会跟进。


## 图神经网络在多模态任务上的应用

### 对话情绪识别（VAT）
&nbsp;&nbsp;&nbsp;&nbsp;这个任务的目的是判断每句话的情绪，数据集是两个人之间的对话，包括 IEMOCAP：151段对话，7433个句子，6种标签，MELD：1433对话，13708句子，304个不同的speaker，7种标签。EmoryNLP。DailyDialog。

1. **MMGCN: Multimodal Fusion via Deep Graph Convolution Network for
Emotion Recognition in Conversation**.  *Jingwen Hu*. ***ACL 2021*** [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11782)][[code](https://github.com/hujingwen6666/MMGCN)] 

    &nbsp;&nbsp;一句话的三个模态作为三个节点，设计了类内边和类外边，是简单的异构图。单独提取了说话者的特征。图边的建模较简单。

2. **MM-DFN: MULTIMODAL DYNAMIC FUSION NETWORK FOR EMOTION RECOGNITION IN CONVERSATIONS**.  *Dou Hu（ 平安保险）*. ***ICASSP 2022*** [[pdf](https://browse.arxiv.org/pdf/2203.02385.pdf)][[code](https://github.com/zerohd4869/MM-DFN)] 

    &nbsp;&nbsp;作者认为之前在这个任务的图神经网络在每一层都积累了冗余信息，限制了模态之间的上下文理解。作者设计了一个有门控的图卷积网络
    
3. **Structure Aware Multi-Graph Network for Multi-Modal Emotion Recognition in Conversations**.  *Duzhen Zhang（中国科学院大学）*. ***TMM 2023*** [[pdf](https://ieeexplore.ieee.org/document/10219015)]

    &nbsp;&nbsp;构建了多个特定于模态的图来模拟多模态上下文的异质性；构造了一个模块来决定语句中有边，该模块通过强制每个话语关注有助于其情感识别的上下文话语来减少冗余，就像消息传播减少器以减轻过度平滑；Dual-Stream Propagation (DSP)构造双流（模态内和模态间）来聚集多个异构模态，包含了门控单元

4. **DualGATs: Dual Graph Attention Networks for
Emotion Recognition in Conversations**.  *Duzhen Zhang（中国科学院大学）*. ***ACL 2023*** [[pdf](https://browse.arxiv.org/pdf/2203.02385.pdf)]

    &nbsp;&nbsp;没仔细看，是第三篇同一个作者。后续跟进。issue看了下有点不想搞。



    
### 假新闻假消息检测（VT）
1. 


### 睡眠状态检测（生理信号）
1. **Jumping Knowledge Based Spatial-Temporal Graph Convolutional Networks for Automatic Sleep Stage Classification**.  *Xiaopeng Ji，Peng Wen*. ***IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2022*** [[pdf](https://ieeexplore.ieee.org/abstract/document/9777906)]

    &nbsp;&nbsp;用jk-net

### 抑郁症（问卷、音频、视频、生理信号）
DAIC-WOZ（问卷、音频、视频、），MODMA（音频+EEG）
1. **HCAG: A Hierarchical Context-Aware Graph Attention Model for Depression Detection**.  *Meng Niu；Lufeng Yang*. ***IICASSP 2021*** [[pdf](https://ieeexplore.ieee.org/document/9413486)]

    &nbsp;&nbsp;根据问答形式将一个问答对的多模态信息聚合，作为一个点，因此最后一个subject是一个图，里面有n个问答对的点，然后用一个readout读取全图信息，判断subject是不是抑郁症。图神经网络选用 GAT（同构图），模态是 A+T。这里一个图的节点数不多。

2. **MS2 -GNN: Exploring GNN-Based Multimodal Fusion Network for Depression Detection**.  *Hu bin（不用多说了） ，Tao Chen*. ***TCYB 2022*** [[pdf](https://ieeexplore.ieee.org/document/9911215)]

    &nbsp;&nbsp;认为患病 subject 之间有共性，因此用不同subject作为节点；使用GCN，因为用到了邻接矩阵。损失函数用的很细致，用了KL散度，类内和类外不同处理的损失。用了 EEG 和音频模态

### 阿尔兹海默症

数据集主要是 ANDI，UKD。主要用 MRI 磁共振成像。之所以是多模态是因为 MRI 包括 

(1)    fMRI，可以对脑部血管氧含量做4D的追踪，氧含量越高代表该部位的脑部运动越强，但是分辨率低


(2）   sMRI， 空间分辨率高，静态解剖3D图,没有时间分辨率，原理是各组织含水量不同。可以很好地看到大脑的细微结构

(3）   dMRI，可以看到神经结构，原理是各组织水分子扩散性质不同

(4） 在肿瘤分割领域还有其他许多模态

    a.  fluid attenuation inversion recovery (FLAIR)

    b.  T1-weighted (T1)

    c.  contrast enhanced T1-weighted (T1ce) 

    d.  T2-weighted (T2)


后续再去了解

1. **Graph Transformer Geometric Learning of Brain Networks Using Multimodal MR Images for Brain Age Estimation**.  *Hongjie Cai，Manhua Liu（上交深耕这个领域的专家）*. ***TMI  2023*** [[pdf](https://ieeexplore.ieee.org/document/9950299)]

    &nbsp;&nbsp;脑区天然有连接关系，用卷积自编码器编码不同模态的表征；基于空间距离、特征相似度、模态相似度构造不同的边，排序前K个认为是有连接，因此是异构图；图像的预处理很重要；用了UKD做预训练，然后用ANDI做微调
2. **基于 MRI 与 SNP 的多模态 AD 早期诊断模型研究** *陈国斌，曾安（广工计院副院长，这个领域的专家）*[[pdf](https://kns.cnki.net/kcms2/article/abstract?v=v0gKrRoz1UfykYPLJuA_MOAEOcRnzrsUbgzqG3-fCH4NLrExxcg3Rz_xAxjyOBAn82YAnTw8qa1YzgpmkYHH_ubv1U-fqiciEKfjF1dOhQDcRG0YN_1L6Q==&uniplatform=NZKPT&language=gb)]
### 帕金森
帕金森症状是手抖等，原因是神经受损导致肢体无法控制。用步态骤停判断。模态是视频和步压。用光流法可以再拓展两个模态。
1.  **Graph Fusion Network-Based Multimodal Learning for Freezing of Gait Detection**.  *Kun Hu，Ah Chung Tsoi, and Simon J. G. Lewis *. ***TNNLS  2021*** [[pdf](https://ieeexplore.ieee.org/abstract/document/9525818)]

    &nbsp;&nbsp;悉尼大学的项目，自己造了个数据集但是好像没开源？作者后续还有发文章。这里把四个模态当成一个图，t 时间 t 个图。根据全图表征判断病人是否有步态骤停。


