import pandas as pd

import torch

from torch_geometric.data import HeteroData


#请提前将net_dbis.zip解压到folder文件夹下
#即folder文件夹下应有叫net_dbis的文件夹，内含id_author.txt id_conf.txt paper_author.txt paper_conf.txt paper.txt
folder='/data/wanghuijuan/hgnn_data'
files_folder=folder+'/net_dbis'

#以下代码参考PyG.datasets.AMiner

data=HeteroData()

path=files_folder+'/id_author.txt'
with open(path,encoding='ISO-8859-1') as f:
    z=f.readlines()
data['author'].num_nodes=len(z)
author_map={z[x].strip().split('\t')[0]:x for x in range(len(z))}

path=files_folder+'/id_conf.txt'
with open(path,encoding='ISO-8859-1') as f:
    z=f.readlines()
data['conf'].num_nodes=len(z)
conf_map={z[x].strip().split('\t')[0]:x for x in range(len(z))}

path=files_folder+'/paper.txt'
with open(path,encoding='ISO-8859-1') as f:
    z=f.readlines()
data['paper'].num_nodes=len(z)
pair=[x.strip().split('  ') for x in z]  #factor eg: ['21525', '', '', ' Methods and Tools for Data Value Re-Engineering.']
data['paper'].raw_feature=[x[-1] for x in pair]
paper_map={pair[x][0]:x for x in range(len(pair))}

path=files_folder+'/paper_author.txt'
with open(path,encoding='ISO-8859-1') as f:
    z=f.readlines()
pair=[x.strip().split('\t') for x in z] 
data['paper','written_by','author'].edge_index=torch.tensor([[paper_map[x[0]],author_map[x[1]]] for x in pair]).T

path=files_folder+'/paper_conf.txt'
with open(path,encoding='ISO-8859-1') as f:
    z=f.readlines()
pair=[x.strip().split('\t') for x in z] 
data['paper','published_in','conf'].edge_index=torch.tensor([[paper_map[x[0]],conf_map[x[1]]] for x in pair]).T

print(data)