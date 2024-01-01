本项目用于整理异质图神经网络（HGNN）相关数据、算法、其他资料。  
非使用GNN方法的异质图相关工作（尤其是推荐系统和知识图谱相关工作）暂不列入。暂时主要考虑创新点在GNN方面的（如NLP领域主要创新点在如何建图方面的，暂不列入）。  
作者主要用PyTorch和PyG。

超过5M的文件都存储在了百度网盘上，以方便大陆用户下载：  
链接：https://pan.baidu.com/s/1P1X9LjT85DU9OBPAr6l3bg  
提取码：ea7u

* [1. 数据](#数据)
* [2. 论文](#论文)
* [3. 工具](#工具)

# 数据
简单介绍：
|**数据集名称**|**下载和预处理方式**|**出处**|**常用任务**|
|---|---|-----|---|
|ogbn-mag|PyG OGM_MAG<br>ogb|MAG数据集的子集|节点分类|
|MAG|已停用|Microsoft academic graph: When experts are not enough.|
|AMiner (metapath2vec)|PyG（百度网盘）|metapath2vec: Scalable Representation Learning for Heterogeneous Networks|节点分类|
|AMinerNetwork|https://www.aminer.org/aminernetwork|A multilayered approach for link prediction in heterogeneous complex networks|链路预测|
|DBIS|https://ericdongyx.github.io/metapath2vec/m2v.html （百度网盘和HGNN_Collection/load_data/dbis_pyg.py）|metapath2vec: Scalable Representation Learning for Heterogeneous Networks|节点相似度
|DBLP (MAGNN)|PyG (DBLP和HGBDataset)（百度网盘）|MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding|节点分类|
|IMDB (MAGNN)|PyG（百度网盘）|MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding|节点分类|
|IMDB (Simple-HGN)|PyG HGBDataset|Are We Really Making Much Progress? Revisiting, Benchmarking, and Refining Heterogeneous Graph Neural Networks|节点分类
|LastFM|PyG（百度网盘）|MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding|链路预测|
|MovieLens (PyG)|PyG|https://movielens.org/|
|ACM|PyG HGBDataset|Are We Really Making Much Progress? Revisiting, Benchmarking, and Refining Heterogeneous Graph Neural Networks|节点分类
|Freebase|PyG HGBDataset|Are We Really Making Much Progress? Revisiting, Benchmarking, and Refining Heterogeneous Graph Neural Networks|节点分类
|ogbl-biokg|https://ogb.stanford.edu/docs/linkprop/||链路预测

详细介绍：  
[《HGNN图数据集》](https://shimo.im/sheets/pmkxQwlp4BumP8AN/MODOC/)

相关介绍博文：
[异质图神经网络（HGNN）常用数据集信息统计（持续更新ing...）_诸神缄默不语的博客-CSDN博客_异质图数据集](https://blog.csdn.net/PolarisRisingWar/article/details/126980733)
[PyG (PyTorch Geometric) Dropbox系图数据集无法下载的解决方案（AMiner, DBLP, IMDB, LastFM）（持续更新ing...）](https://blog.csdn.net/PolarisRisingWar/article/details/126980943)

# 论文
**2023年**  
通用节点嵌入：
1. (WWW) [A Post-Training Framework for Improving Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2304.00698)
2. (Journal of Computational Science) [Multi-view contrastive learning for multilayer network embedding](https://www.sciencedirect.com/science/article/abs/pii/S1877750323000352)
2. (Entropy) [Unsupervised Embedding Learning for Large-Scale Heterogeneous Networks Based on Metapath Graph Sampling](https://www.mdpi.com/1099-4300/25/2/297)
4. (Scientific Reports) [A multi-view contrastive learning for heterogeneous network embedding](https://www.nature.com/articles/s41598-023-33324-7)
2. [HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer](https://arxiv.org/abs/2302.11329)
2. [Heterophily-Aware Graph Attention Network](https://arxiv.org/abs/2302.03228)

动态图节点嵌入：
1. (Neurocomputing) [Dynamic heterogeneous graph representation learning with neighborhood type modeling](https://www.sciencedirect.com/science/article/abs/pii/S0925231223002096)

graph rewiring:
1. (WWW) [Homophily-oriented Heterogeneous Graph Rewiring](https://arxiv.org/abs/2302.06299)：同配性+异质性

graphlet和orbit计数：
1. [Graphlet and Orbit Computation on Heterogeneous Graphs](https://arxiv.org/abs/2304.14268)

交叉学科：
1. (Phys. Rev. E) [Deterministic, quenched and annealed parameter estimation for heterogeneous network models](https://arxiv.org/abs/2303.02716)：计量经济学+物理

**2022年**  
综述：
1. (Artificial Intelligence Review) [Heterogeneous graph neural networks analysis: a survey of techniques, evaluations and applications](https://link.springer.com/article/10.1007/s10462-022-10375-2)

通用节点嵌入：
1. (World Wide Web) [HGNN-ETA: Heterogeneous graph neural network enriched with text attribute](https://link.springer.com/article/10.1007/s11280-022-01120-4)：利用节点的文本特征信息
6. (World Wide Web) [How the four-nodes motifs work in heterogeneous node representation?](https://link.springer.com/article/10.1007/s11280-022-01115-1)
3. (CIKM) [SplitGNN: Splitting GNN for Node Classification with Heterogeneous Attention](https://arxiv.org/abs/2301.12885)：联邦学习
2. (SDM) [Structure-Enhanced Heterogeneous Graph Contrastive Learning](https://www.cs.emory.edu/~jyang71/files/stencil.pdf)：提出STENCIL模型，跨视图（保证视图之间一致性最大化）（基于metapaths构建视图（metapath实例起终点构成的同质图），最大化同一节点在不同视图上嵌入的相似性，将各视图的嵌入attentively聚合）+对比学习+结构嵌入
2. (Transactions on Big Data) [A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources](https://arxiv.org/abs/2011.14867)：综述
2. (Knowledge-Based Systems) Megnn: Meta-path extracted graph neural network for heterogeneous graph representation learning：自动提取metapaths+可解释性
2. (IEEE Transactions on Knowledge and Data Engineering) Heterogeneous Graph Representation Learning with Relation Awareness：提出R-HGNN模型
3. (IEEE Transactions on Knowledge and Data Engineering) mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding：meta graph
4. (IEEE Transactions on Knowledge and Data Engineering) RHINE: Relation Structure-Aware Heterogeneous Information Network Embedding
5. (IEEE Transactions on Knowledge and Data Engineering) Explicit Message-Passing Heterogeneous Graph Neural Network：提出EMP模型。本文认为传统用metapaths将异质图构建为同质图的方法是隐式的，而EMP模型显式信息传递
19. (南洋理工大学硕士论文) [Contrastive learning for heterogeneous graph neural networks](https://dr.ntu.edu.sg/handle/10356/164475)
5. (Neural Networks) Latent neighborhood-based heterogeneous graph representation：随机游走增强邻居+GTN学习metapaths
6. (IEEE Transactions on Neural Networks and Learning Systems) Learning Knowledge Graph Embedding With Heterogeneous Relation Attention Networks
7. (Appl. Sci.) MBHAN: Motif-Based Heterogeneous Graph Attention Network：基于motif
8. (Data Mining and Knowledge Discovery) Personalised meta-path generation for heterogeneous graph neural networks：提出PM-HGNN模型，强化学习（将找metapaths视作马尔科夫决策过程）
9. (Machine Learning) [Heterogeneous graph embedding with single-level aggregation and infomax encoding](https://link.springer.com/article/10.1007/s10994-022-06160-5)：提出无监督HGNN模型HIME，对节点特征应用MLP（每一种节点用一个模型），直接进行聚合。损失函数鼓励邻居相近+信息最大化
10. (International Journal of Machine Learning and Cybernetics) [Multiple heterogeneous network representation learning based on multi-granularity fusion](https://link.springer.com/article/10.1007/s13042-022-01665-w)：将结构（一阶邻居）和语义（meta-path）视作不同粒度，结合来学习表征
11. (Journal of Applied Mathematics) [Classification Algorithm for Heterogeneous Network Data Streams Based on Big Data Active Learning](https://downloads.hindawi.com/journals/jam/2022/2996725.pdf)
9. (IEEE Access) Siamese Network Based Multiscale Self-Supervised Heterogeneous Graph Representation Learning：提出SNMH模型，自监督学习+对比学习（metapaths和one-hop）+孪生神经网络
18. (ICCWAMTIP) [Relation Heterogeneous Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/10016506)
21. (ICDMW) [Graph Convolutional Neural Network based on the Combination of Multiple Heterogeneous Graphs](https://ieeexplore.ieee.org/abstract/document/10031027)
6. Simple and Efficient Heterogeneous Graph Neural Network：提出SeHGNN模型，预处理+无参+轻量级
7. Relation Embedding based Graph Neural Networks for Handling Heterogeneous Graph：不用metapaths
8. Descent Steps of a Relation-Aware Energy Produce Heterogeneous Graph Neural Networks：关注过平滑问题
9. Heterogeneous Graph Neural Networks using Self-supervised Reciprocally Contrastive Learning：提出HGCL模型，自监督学习+对比学习（节点属性和拓扑结构）
10. Heterogeneous Graph Contrastive Multi-view Learning：提出HGCML模型，关注减轻对比学习（metapaths）中的采样偏差
10. Heterogeneous Graph Masked Autoencoders：提出HGMAE模型，auto encoder
11. [Meta-node: A Concise Approach to Effectively Learn Complex Relationships in Heterogeneous Graphs](https://arxiv.org/abs/2210.14480)：不用预定义的meta-paths或meta-graphs

预训练：
1. (NeurIPS) [Self-supervised Heterogeneous Graph Pre-training Based on Structural Clustering](https://arxiv.org/abs/2210.10462)

multiplex graph：
1. (KDD) Multiplex Heterogeneous Graph Convolutional Network
2. (ijtr) [DSMN: A New Approach for Link Prediction in Multilplex Networks](https://ijict.itrc.ac.ir/browse.php?a_code=A-10-3655-1&slc_lang=other&sid=1)

小样本学习：
1. (KDD) Few-shot Heterogeneous Graph Learning via Cross-domain Knowledge Transfer

AI安全：
1. (AAAI) Robust Heterogeneous Graph Neural Networks against Adversarial Attacks

知识蒸馏：
1. (WWW) Collaborative Knowledge Distillation for Heterogeneous Information Network Embedding
2. (Neurocomputing) HIRE: Distilling high-order relational knowledge from heterogeneous graph neural networks

匹配节点：
1. (AAAI) From One to All: Learning to Match Heterogeneous and Partially Overlapped Graphs

链路预测：
1. (ICDM) [Revisiting Link Prediction on Heterogeneous Graphs with a Multi-view Perspective](https://ieeexplore.ieee.org/abstract/document/10027638)

**2021年**  
通用节点嵌入：
1. (KDD) Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks：先把近年多篇论文（HAN、GTN、RSHN、HetGNN、MAGNN、HGT、HetSANN、RGCN、GATNE、KGCN、KAGT）喷了一遍，然后提出Simple-HGN模型（参考博文：[Re10：读论文 Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous gr_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/126009977)）
2. (KDD) [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://arxiv.org/abs/2105.09111)：提出HeCo模型，跨视图（metapath和network）+对比学习+自监督学习
3. (KDD) HGK-GNN: Heterogeneous Graph Kernel based Graph Neural Networks：引入HGK
3. (KDD) DiffMG: Differentiable Meta Graph Search for Heterogeneous Graph Neural Networks：NAS
2. (AAAI) Heterogeneous graph structure learning for graph neural networks
3. (WWW) Heterogeneous Graph Neural Network via Attribute Completion：提出HGNN-AC模型，端到端地同时学习节点特征补全和GNN学习
4. (IJCAI) Heterogeneous Graph Information Bottleneck：提出HGIB模型，无监督学习+信息论
5. (ECML PKDD) Multi-view Self-supervised Heterogeneous Graph Embedding：提出MVSE模型，用基于metapaths的多视图实现自监督学习
3. (IEEE Transactions on Knowledge and Data Engineering) Heterogeneous Graph Propagation Network：提出HPN模型，缓解HGNN中的confusion phenomenon问题（类似同质GNN中的过平滑问题）
5. (IEEE Transactions on Knowledge and Data Engineering) Interpretable and Efficient Heterogeneous Graph Convolutional Network：提出ie-HGCN模型，可以广泛评估各种metapaths
6. (IEEE Transactions on Knowledge and Data Engineering) Higher-Order Attribute-Enhancing Heterogeneous Graph Neural Networks：提出HAE模型，使用了meta-graphs
7. (IEEE Transactions on Knowledge and Data Engineering) HGATE: Heterogeneous Graph Attention Auto-Encoders：无监督学习+auto encoder
8. (IEEE Transactions on Knowledge and Data Engineering) Walking with Attention: Self-guided Walking for Heterogeneous Graph Embedding：提出SILK模型，基于随机游走
7. (Big Data Mining and Analytics) Attention-aware heterogeneous graph neural network：提出AHNN模型，metapath+attention
8. (ICME) Revisiting Graph Neural Networks for Node Classification in Heterogeneous Graphs：关注过拟合问题
8. (ICLR 2022被拒+撤回投稿) R-GSN: The Relation-based Graph Similar Network for Heterogeneous Graph：不用metapaths，据称在ogbn-mag图上达到了SOTA效果

动态图：
1. (ECML PKDD) Dynamic Heterogeneous Graph Embedding via Heterogeneous Hawkes Process：霍克斯过程

超图：
1. (WSDM) Heterogeneous Hypergraph Embedding for Graph Classification

社区检测：
1. (CIKM) Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model：提出CP-GNN模型

图生成：
1. (ICDM) Deep Generation of Heterogeneous Networks：提出HGEN模型

**2020年**  
通用节点嵌入：
1. (IJCAI) Heterogeneous Network Representation Learning：综述
2. (AAAI) An Attention-based Graph Neural Network for Heterogeneous Structural Learning：提出HetSANN模型，基于关系类型，建立attention机制，实现节点信息聚合，不使用metapath（参考博文：[Re22：读论文 HetSANN An Attention-based Graph Neural Network for Heterogeneous Structural Learning_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/126058473)）
2. (WWW) [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/abs/2002.01680)：首先对节点特征进行转换，然后聚合metapath内部信息，然后聚合各metapath的信息
3. (WWW) [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332)：提出HGT模型，把整个Transformer结构改到图上，这种感觉
4. Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning：提出HGConv模型
5. Reinforcement Learning Enhanced Heterogeneous Graph Neural Network：提出HGRL模型，强化学习（将找metapaths视作马尔科夫决策过程）
6. (被ICLR 2021拒了) Scalable Graph Neural Networks for Heterogeneous Graphs：提出NARS模型，关注scalability问题

动态图：
1. (ECIR) Dynamic Heterogeneous Graph Embedding Using Hierarchical Attentions：提出DyHAN模型（实验仅做了链路预测）
2. (ICKG) Heterogeneous Dynamic Graph Attention Network：提出HDGAN模型
2. Meta Graph Attention on Heterogeneous Graph with Node-Edge Co-evolution：提出CoMGNN和ST-CoMGNN模型

multiplex graph：
1. (AAAI) [Unsupervised Attributed Multiplex Network Embedding](https://arxiv.org/abs/1911.06750)：提出DMGI模型

多模态：
1. (KDD) HGMF: Heterogeneous Graph-based Fusion for Multimodal Data with Incompleteness

anchor link prediction任务：
1. (AAAI) Type-Aware Anchor Link Prediction across Heterogeneous Networks Based on Graph Attention Network

**2019年**  
通用节点嵌入：
1. (WWW) [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293)：提出HAN模型，先基于metapath attentively聚合节点信息，然后attentively聚合metapath信息
2. (NeurIPS) [Graph Transformer Networks](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)：提出GTN模型，自动学习metapaths
3. (KDD) [Heterogeneous Graph Neural Network](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961)：提出HetGNN模型，用RWR抽样异质邻居，按节点类型分类，然后用聚合
4. (KDD) [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669)：提出GATNE模型
5. (KDD) [Adversarial Learning on Heterogeneous Information Networks](http://shichuan.org/doc/70.pdf)：提出HetGAN模型
3. (ICDM) Relation Structure-Aware Heterogeneous Graph Neural Network：提出RSHN模型，用coarsened line graph先获得边特征，然后传播节点和边特征
4. (CIKM) [BHIN2vec: Balancing the Type of Relation in Heterogeneous Information Network](https://arxiv.org/abs/1912.08925)：提出BHIN2vec模型，随机游走+skip gram+多任务，解决HIN中不同种类边数不平衡的问题
6. [Heterogeneous Deep Graph Infomax](https://arxiv.org/abs/1911.08538v5)：提出HDGI模型，信息论+无监督学习

图匹配：
1. (IJCAI) Heterogeneous Graph Matching Networks for Unknown Malware Detection

底层编译技术：
1. (PACT) Gluon-Async: A Bulk-Asynchronous System for Distributed and Heterogeneous Graph Analytics

**2018年**  
通用节点嵌入：
1. (CIKM) Are Meta-Paths Necessary?: Revisiting Heterogeneous Graph Embeddings
2. (KDD) [Easing Embedding Learning by Comprehensive Transcription of Heterogeneous Information Networks](https://dl.acm.org/doi/10.1145/3219819.3220006)
3. (TKDE) [Heterogeneous Information Network Embedding for Recommendation](https://arxiv.org/abs/1711.10730)：提出HERec模型，将异质图转换为metapath-based graphs，用skip-gram嵌入
2. (ESWC) [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)：提出RGCN模型
3. (SIAM) [AspEm: Embedding Learning by Aspects in Heterogeneous Information Networks](https://arxiv.org/abs/1803.01848)

链路预测：
1. (KDD) [PME: Projected Metric Embedding on Heterogeneous Networks for Link Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/pme-projected-metric-embedding-on-heterogeneous-networks-for-link-predictio)

**2017年**  
通用节点嵌入：
1. (KDD) [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)：用基于metapath的随机游走来构建邻居，然后用类似word2vec的逻辑来实现节点表征（类似DeepWalk）（参考博文：[Re31：读论文 metapath2vec: Scalable Representation Learning for Heterogeneous Networks_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/127055716)）
2. (KDD) [Meta-Path Guided Embedding for Similarity Search in Large-Scale Heterogeneous Information Networks](https://arxiv.org/abs/1610.09769)：提出ESim模型，根据预定义的权重从抽样出的metapaths实例中捕获节点语义
2. (CIKM) [HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning](https://dl.acm.org/doi/pdf/10.1145/3132847.3132953)
3. (WSDM) [Embedding of Embedding (EOE): Joint Embedding for Coupled Heterogeneous Networks](http://shichuan.org/hin/time/2017.%20WSDM%20Embedding%20of%20Embedding%20EOE%20Joint%20Embedding%20for%20Coupled%20Heterogeneous%20Networks.pdf)

**2015年**  
1. (KDD) [PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks](https://arxiv.org/abs/1508.00200)

**2014年**  
1. (KDD) [Mining heterogeneous information networks: a structural analysis approach](https://www.kdd.org/exploration_files/V14-02-03-Sun.pdf)
2. (CIKM) [Meta-Path-Based Ranking with Pseudo Relevance Feedback on Heterogeneous Graph for Citation Recommendation](https://dl.acm.org/doi/10.1145/2661829.2661965)

**2013年**  
博弈论：
1. (PLoS ONE) Evolution of Cooperation in a Heterogeneous Graph: Fixation Probabilities under Weak Selection

**2011年**  
节点分类：
1. (World Wide Web) Graffiti: graph-based classification in heterogeneous networks

节点相似性：
1. (Proceedings of the VLDB Endowment) [PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks](http://www.vldb.org/pvldb/vol4/p992-sun.pdf)

**2010年**  
通用节点嵌入：
1. (ECML PKDD) Graph Regularized Transductive Classification on Heterogeneous Information Networks
2. Optimal Embedding of Heterogeneous Graph Data with Edge Crossing Constraints 

# 工具
1. [BUPT-GAMMA/OpenHINE: An Open-Source Toolkit for Heterogeneous Information Network Embedding (HINE)](https://github.com/BUPT-GAMMA/OpenHINE)