from argparse import Namespace
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

from deepcancer.fpgnn.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from deepcancer.fpgnn.features import GetPubChemFPs
from deepcancer.fpgnn.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule. 消息传递神经网络，用于编码分子。"""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        """
        初始化MPNEncoder。

         ：param args：参数。
         ：param atom_fdim：Atom功能尺寸。
         ：param bond_fdim：边特征尺寸。
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size#300,args已定义
        self.bias = args.bias#false图层不会学习附加偏差
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected#false
        self.atom_messages = args.atom_messages #原是false，此时是true
        self.features_only = args.features_only #false
        self.use_input_features = args.use_input_features#None
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        """
        torch.nn.Parameter(torch.FloatTensor(hidden_size)),看了官方教程里面的解释也是云里雾里，于是在栈溢网看到了一篇解释，并做了几个实验才算完全理解了这个函数。
        首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor，转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化


        """

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        #if self.atom_messages ：input_dim = self.atom_fdim else input_dim = self.bond_fdim   普通运行下是boom_fdim，此处是atom
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        #dropout_layer   act_func  W_i  w_h_  W_o  都是网络层

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
         :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        
        编码一批分子图。

         ：param mol_graph：一个BatchMolGraph代表一批分子图。
          ：param features_batch：包含其他功能的ndarray列表。
         ：return：一个PyTorch张量，形状为（num_molecules，hidden_size），其中包含每个分子的编码。
         
        """
        if self.use_input_features:#none
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()  #列表
        #print('f_atoms.size():',f_atoms.size()) torch.Size([1320, 133]) torch.Size([1320, 133]) ... torch.Size([1231, 133])
        #应该是（50个分子的原子个数）x原子维度

        #数据送进cuda
        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size  用原子信息进行训练
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size  用边信息进行训练
        #以上是W*cat(Xv,evw)；以下是relu(W*cat(Xv,evw)).结果是hvw0
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):#depth=3
            if self.undirected:#False
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden   index_select_ND从对应于index中的原子或键索引的源中选择消息特性。
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)
        #print('MPNEncoder-mol_vecs.size():',mol_vecs.size()) torch.Size([50, 300])
        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule. 消息传递神经网络，用于编码分子"""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        """
        初始化MPN。

         ：param args：参数。
         ：param atom_fdim：Atom功能尺寸。
         ：param bond_fdim：绑定特征尺寸。
         ：param graph_input：如果为true，则将BatchMolGraph作为输入。 否则，将使用微笑字符串列表作为输入。
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        #print('!!!!!~~~~!!!!!    graph_input= ',graph_input)
        #运行时graph_input=False 
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        """
        对一批分子SMILES字符串进行编码。

         ：param batch：SMILES字符串或BatchMolGraph的列表（如果self.graph_input为True）。
         ：param features_batch：包含其他功能的ndarray列表。
         ：return：包含每个分子编码的形状（num_molecules，hidden_size）的PyTorch张量。
        """
        if not self.graph_input:  # if features only, batch won't even be used 如果仅功能，则甚至不会使用批处理   graph_input=False if会执行
            batch = mol2graph(batch, self.args) #将smiles处理成图向量

        output = self.encoder.forward(batch, features_batch)
        #print('MPN-output.size()',output.size()) torch.Size([50, 300])
        return output


class FPN(nn.Module):
    def __init__(self, args:Namespace):
    #输入应该是smiles字符串
        super(FPN, self).__init__()
        self.fp_2_dim=args.fp_2_dim#指纹1489->fp_2_dim->300
        self.args = args
        self.dropout_fpn=args.dropout
        #self.fc1=nn.Linear(1489, 512)
        self.fc1=nn.Linear(1489, self.fp_2_dim)
        self.act_func = get_activation_function(args.activation)
        #self.fc2=nn.Linear(512, 300)
        self.fc2=nn.Linear(self.fp_2_dim, 300)
        self.dropout=nn.Dropout(p=self.dropout_fpn)
    
    def forward(self, x):
        """
        print('!!!!!!!!!!!!!!!!!!!!!!')
        print('打印FPN.forward.x')
        print('!!!!!!!!!!!!!!!!!!!!!!')
        print(x) x是一组smiles([smile1,smile2,...,smile50],None) [smile,...]是model(batch,fetures_batch)里的batch  Nones是fetures_batch
        print('!!!!!!!!!!!!!!!!!!!!!!')
        print('打印FPN.forward.x结束')
        print('!!!!!!!!!!!!!!!!!!!!!!')
        """

        fp_list=[]
        smil = x[0]
        #smiles转为指纹fp
        for i, smi in enumerate(smil):
            fp=[]
            #print('i',i)
            #print('smi',smi)
            smiles = str(smi)
            mol = Chem.MolFromSmiles(smiles)
            #fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)#Morgan指纹 1024bit
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)#MACCS指纹 167bit
            fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1) #441bit
            fp_pubcfp = GetPubChemFPs(mol)#pubchemfp 881bit
            
            #fp.extend(fp_morgan)
            fp.extend(fp_maccs)
            fp.extend(fp_phaErGfp)
            fp.extend(fp_pubcfp)
            #print('fp:',fp)
            fp_list.append(fp)
            #print('!!!!fp_list=',fp_list) #fp_list就是这组smiles的fp
            
        fp_list=torch.Tensor(fp_list)
        #print('fp_list.size:',fp_list.size())
        #print('fp_list.size()',fp_list.size()) torch.Size([50, 1024])
        fp_list=fp_list.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #print('fp_list.device:',fp_list.device)
        output1=self.fc1(fp_list)
        #print('output1:',output1)
        output1=self.dropout(output1)
        output1=self.act_func(output1)
        output1=self.fc2(output1)
        #print('output2:',output1)
        return output1#fp_list#output

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features #输入h的维度
        self.out_features = out_features#输出h'的维度
        self.alpha = alpha #leakrelu参数 0.2
        self.concat = concat #True

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))#定义W是模型的参数
        nn.init.xavier_uniform_(self.W.data, gain=1.414)#给W随机初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W) # shape [N, out_features]   torch.mm是相乘
        N = h.size()[0]#N是节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        #tensor.repeat 复制函数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)#adj>0即有连接处注意力是e，无连接处注意力负无穷大
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)#dropout=0.6
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime #N x out 即Nx300维


class GATOneMol(nn.Module):
    def __init__(self,args: Namespace):
        super(GATOneMol, self).__init__()
        """
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        self.nfeat=133#每个原子133维
        #self.nhid=60
        self.nhid=args.nhid
        #self.dropout=0.6
        self.dropout=args.dropout_gat
        self.nclass=300
        self.alpha=0.2
        self.nheads=args.nheads
        #self.nheads=8
        self.args = args

        self.attentions = [GraphAttentionLayer(self.nfeat, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.nclass, dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, x, adj):#x 输入h adj领接矩阵
        x = F.dropout(x, self.dropout, training=self.training)
        #print('spGAT-x1.size():',x.size())#N x atom_feature  torch.Size([27, 133])
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #print('spGAT-x2.size():',x.size())#N x (nhid*nheads) torch.Size([27, 480])
        x = F.dropout(x, self.dropout, training=self.training)
        #print('spGAT-x3.size():',x.size())# torch.Size([27, 480])
        x = F.elu(self.out_att(x, adj))
        #print('spGAT-x4.size():',x.size())#N x nclass  torch.Size([27, 300])
        #print('x:',x)
        return F.log_softmax(x, dim=1)


class GATEncoder(nn.Module):
    def __init__(self,args: Namespace):
        super(GATEncoder,self).__init__()
        self.hidden_size = args.hidden_size#300,args已定义
        self.args = args
        self.encoder=GATOneMol(self.args)
    
    
    def forward(self,mol_graph,features_batch,smi_batch):
        
        #f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope  = mol_graph.get_components()  #列表
        f_atoms, a_scope = mol_graph.get_components()
        #print('a_scope',a_scope)
        #print('f_atoms.size():',f_atoms.size()) torch.Size([1320, 133]) torch.Size([1320, 133]) ... torch.Size([1231, 133])
        #应该是（50个分子的原子个数）x原子维度
        f_atoms=f_atoms.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        gat_outs=[]
        for i,smile in enumerate(smi_batch):#smile是一个smile
            x=[]
            adj=[]
            mol = Chem.MolFromSmiles(smile)
            adj=Chem.rdmolops.GetAdjacencyMatrix(mol)#领接矩阵
            adj=adj/1 #艹，adj变成int
            
            adj=torch.from_numpy(adj)
            adj=adj.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            a_start, a_size=a_scope[i]
            
            """
            检查是否在gpu上跑
            """
            #print('MODEL is cuda?:',next(GATOneMol.parameters(self)).is_cuda) True
            #print('adj.device:',adj.device)
            for j in range(a_size):
                x.append(f_atoms[a_start+j])
            x=torch.Tensor([item.cpu().detach().numpy() for item in x]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            #print('x.device:',x.device)
            #print('GATEncoder-x.size():',x.size()) torch.Size([28, 133])
            #print('GATEncoder-x内容:',x)
            #print('GATEncoder-a_size:',a_size) 28
            gat_atoms_out=self.encoder(x,adj)#N x 300
            gat_out=gat_atoms_out.sum(dim=0)/a_size
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0)
        #print('gat_outs.device:',gat_outs.device) cuda
        #print('GATEncoder-gat_outs.size()',gat_outs.size())#应为50*300 torch.Size([50, 300])
        return gat_outs

class GAT(nn.Module):
    def __init__(self,args:Namespace,graph_input: bool = False):
        super(GAT,self).__init__()
        self.args=args
        self.graph_input = graph_input
        self.encoder=GATEncoder(self.args)
        
    def forward(self,batch,features_batch):
        if not self.graph_input:  # if features only, batch won't even be used 如果仅功能，则甚至不会使用批处理   graph_input=False if会执行
            batch_mol = mol2graph(batch, self.args) #将smiles处理成图向量

        output = self.encoder.forward(batch_mol, features_batch,batch)

        return output








