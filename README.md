本项目用于整理异质图神经网络（HGNN）相关数据、算法、其他资料。  
非使用GNN方法的异质图相关工作（尤其是推荐系统和知识图谱相关工作）暂不列入。暂时主要考虑创新点在GNN方面的（如NLP领域主要创新点在如何建图方面的，暂不列入）。  
作者主要用PyTorch和PyG。

超过5M的文件都存储在了百度网盘上，以供大陆用户方便下载：  
链接：https://pan.baidu.com/s/1P1X9LjT85DU9OBPAr6l3bg  
提取码：ea7u

* [1. 数据](#数据)
* [2. 论文](#论文)

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
[PyG (PyTorch Geometric) Dropbox系图数据集无法下载的解决方案（AMiner, DBLP, IMDB, LastFM）（持续更新ing...）](https://blog.csdn.net/PolarisRisingWar/article/details/126980943)

# 模型
**2022年**  
通用节点嵌入：
1. (Transactions on Big Data) A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources：综述

**2020年**  
通用节点嵌入：
1. (AAAI) An Attention-based Graph Neural Network for Heterogeneous Structural Learning：提出HetSANN模型，基于关系类型，建立attention机制，实现节点信息聚合，不使用metapath（参考博文：[Re22：读论文 HetSANN An Attention-based Graph Neural Network for Heterogeneous Structural Learning_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/126058473)）

**2017年**  
通用节点嵌入：
1. (KDD) metapath2vec: Scalable Representation Learning for Heterogeneous Networks：用基于metapath的随机游走来构建邻居，然后用类似word2vec的逻辑来实现节点表征（参考博文：[Re31：读论文 metapath2vec: Scalable Representation Learning for Heterogeneous Networks_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/127055716)）

