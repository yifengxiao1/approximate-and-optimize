import os

import torch
import torch_geometric
import gzip
import pickle
import numpy as np
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1 # 边特征数
        var_nfeats = 6  # 变量特征数

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),# 归一化
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()


        self.output_module = torch.nn.Sequential( #输出，接了两个线性输出
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) #沿着一个新维度对输入张量序列进行连接。原来indicate是【约束编号】【非零系数编号】

        # First step: linear embedding layers to a common dimension (64) 先embedding到共同维数
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions 两个半卷积
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    pytorch geometric已经提供了二分图卷积，我们只需要提供所传递消息的确切形式。
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """


        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        b=torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a=self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )


    def message(self, node_features_i, node_features_j, edge_features):
        #node_features_i,the node to be aggregated
        #node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    这个类编码一组图，以及一个从磁盘加载此类图的方法。可以依次使用pytorch geometric提供的数据加载程序
    """

    def __init__(self, sample_files):# 初始化
        super().__init__(root=None, transform=None, pre_transform=None) # super().__init__() 就是调用父类的init方法， 同样可以使用super()去调用父类的其他方法。 ROOT:应保存数据集的根目录;transform（可调用，可选）：一个接受：obj:`torch_geometry.data.data`对象并返回转换版本的函数/转换。数据对象将在每次访问之前进行转换。（默认值：obj:`None`）
        # pre_transform（可调用，可选）：一个函数/转换，接受：obj:`torch_geometry.data.data`对象并返回转换后的版本。数据对象将在保存到磁盘之前进行转换。（默认值：：obj:`None`）
        self.sample_files = sample_files

    def len(self): # 返回数据集中的样本数量
        return len(self.sample_files)


    def process_sample(self,filepath): # 处理数据返回二分图，约束和解（解包数据？）
        BGFilepath, solFilePath = filepath  # 二分图和解
        # 读取
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData # 二分图
        varNames = solData['var_names'] #变量名

        sols = solData['sols'][:50]#[0:300] 解
        objs = solData['objs'][:50]#[0:300] 变量

        sols=np.round(sols,0)  # 取整，第一个参数是要四舍五入的数字，第二个参数（可选）是小数位数或整数位数，表示要保留的小数位数或整数位数，默认为0。
        return BG,sols,objs,varNames


    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        此方法加载数据收集期间保存在磁盘上的节点二分图观测。
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        # 调用process_sample
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index]) # objs解的目标函数值，sols：Pool中特定解的决策变量值

        A, v_map, v_nodes, c_nodes, b_vars=BG # gurobi里面BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]

        constraint_features = c_nodes  # [当前约束系数和/系数均值,当前约束中系数均值,约束右端项,sense]包括目标函数的  表示约束的特征
        edge_indices = A._indices()  # 返回字典中所有的key。 #A: indices_spr第一维放约束编号，第二维放入系数非零的索引, values_spr记录系数不为0的个数,ncons约束数目，nvars获得变量总数

        variable_features = v_nodes  # 节点特征 第一维：[变量编号]，第二维：[变量在约束平均系数,在约束中出现的次数,在约束中系数最大值，在约束中系数最小值]
        edge_features =A._values().unsqueeze(1)  # 返回字典中所有的value。 unsqueeze(1)的作用是在第二个维度（即索引为1的维度）上增加一维，且大小为1。举个例子：假设原始张量为tensor([1, 2, 3])，则unsqueeze(1)的结果为tensor([[1], [2], [3]])。通过增加维度
        edge_features=torch.ones(edge_features.shape) # 创建一个和edge_features同样大小的全1的张量

        constraint_features[np.isnan(constraint_features)] = 1 # 函数用于判断一个数组中的元素是否为NaN（not a number）,把空的替换为1
        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features), # torch.FloatTensor类型转换, 将list ,numpy转化为tensor
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )



        # We must tell pytorch geometric how many nodes there are, for indexing purposes 为了索引，要告诉pytorch geometric有多少个节点
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]  #节点数等于约束数加变量数
        graph.solutions = torch.FloatTensor(sols).reshape(-1) # 改成一串，没有行列

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]  #shape[0]读取矩阵第一维度的长度
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames # 变量名
        varname_dict={}  #变量名字典key:名字，value：编号
        varname_map=[]
        i=0
        for iter in varNames:
            varname_dict[iter]=i
            i+=1
        for iter in v_map:
            varname_map.append(varname_dict[iter])


        varname_map=torch.tensor(varname_map)

        graph.varInds = [[varname_map],[b_vars]]# [变量名编号映射，二元变量编号]

        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    此类以pytorch几何数据处理程序可以理解的格式对“ecole.obstration.NodeBipartite”观察函数返回的节点二分图观察进行编码。
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):# ????
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        我们重载了pytorch几何方法，该方法告诉在连接那些不明显的条目（边索引，候选）的图时如何增加索引。
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]] # x.size(0)返回shape的第0维度
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy_position(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 21

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions 两次半卷积
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output

class GraphDataset_position(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)


    def process_sample(self,filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols'][:50]#[0:300]
        objs = solData['objs'][:50]#[0:300]

        sols=np.round(sols,0)
        return BG,sols,objs,varNames


    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars=BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features =A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)

        lens = variable_features.shape[0]
        feature_widh = 12  # max length 4095
        position = torch.arange(0, lens, 1)

        position_feature = torch.zeros(lens, feature_widh)
        for i in range(len(position_feature)):
            binary = str(bin(position[i]).replace('0b', ''))

            for j in range(len(binary)):
                position_feature[i][j] = int(binary[-(j + 1)])



        #  转到CPU
        constraint_features = constraint_features.cpu()
        edge_indices = edge_indices.cpu()
        edge_features = edge_features.cpu()
        variable_features = variable_features.cpu()

        v = torch.concat([variable_features, position_feature], dim=1)

        variable_features = v
        # constraint_features = constraint_features.to('cpu')
        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict={}
        varname_map=[]
        i=0
        for iter in varNames:
            varname_dict[iter]=i
            i+=1
        for iter in v_map:
            varname_map.append(varname_dict[iter])


        varname_map=torch.tensor(varname_map)

        graph.varInds = [[varname_map],[b_vars]]

        return graph

def postion_get(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 12  # max length 4095
    position = torch.arange(0, lens, 1)

    position_feature = torch.zeros(lens, feature_widh)
    for i in range(len(position_feature)):
        binary = str(bin(position[i]).replace('0b', ''))

        for j in range(len(binary)):
            position_feature[i][j] = int(binary[-(j + 1)])

    variable_features = torch.FloatTensor(variable_features.cpu())
    v = torch.concat([variable_features, position_feature], dim=1).to(DEVICE)
    return v
