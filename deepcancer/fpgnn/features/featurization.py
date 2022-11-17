from argparse import Namespace
from typing import List, Tuple, Union

import torch
from rdkit import Chem

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),#原子类型（例如C，N，O），按原子序数
    'degree': [0, 1, 2, 3, 4, 5],#原子参与的键数
    'formal_charge': [-1, -2, 1, 2, 0],#分配给原子的整数电子电荷
    'chiral_tag': [0, 1, 2, 3],#未指定，四面体CW / CCW或其他
    'num_Hs': [0, 1, 2, 3, 4],#键合氢原子数
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
#len(choices) + 1包含不常见值的空间  
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization 记忆
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.哪位是1
    :param choices: A list of possible values.可能值列表
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.  长度len（choices）+ 1的列表的独热码。
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to. functional_groups：k热码，指示原子所属的官能团。
    :return: A list containing the atom features. 
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features 缩放到与其他功能大致相同的范围
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.

     MolGraph表示单个分子的图结构和特征。

     MolGraph计算以下属性：
     -微笑：微笑字符串。
     -n_atoms：分子中的原子数。
     -n_bonds：分子中的键数。
     -f_atoms：从原子索引到列表原子特征的映射。
     -f_bonds：从键索引到键特征列表的映射。
     -a2b：从原子索引到传入键索引列表的映射。
     -b2a：从键索引到键所起源的原子的索引的映射。
     -b2revb：从键索引到反向键索引的映射。
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.计算分子的图形结构和特征化。

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features 原子索引到原子特征的映射
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features 键索引到（in_atom，键）串联的映射
        self.a2b = []  # mapping from atom index to incoming bond indices 原子索引到传入键索引指标的映射。
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond从键索引到反向键索引的映射。
        #这些都是列表

        # Convert smiles to molecule 将微笑转化为分子
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures 如果我们正在折叠子结构，则假冒“原子”的数量    
        self.n_atoms = mol.GetNumAtoms() #只是得到原子数量
        
        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):#一个个按顺序添加原子特征
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
        """
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):#对每个a1原子 即边的起点v
            for a2 in range(a1 + 1, self.n_atoms):#遍历a1之后的原子，即构架边 a2原子为边终点w
                bond = mol.GetBondBetweenAtoms(a1, a2)#边是否存在 bond此时=0or1

                if bond is None:#如果边不存在，那么跳过
                    continue

                f_bond = bond_features(bond)#如果边存在，那么给出边的特征

                if args.atom_messages:#atom_messages=1 用点信息迭代 =0 用边信息迭代
                    self.f_bonds.append(f_bond)#=1，用点信息迭代，初始边信息为边信息，且双倍添加，因为是有向边
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)#=0，用边信息迭代，边信息=起点+边
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds  #边的数量or去边的索引 初始=0
                b2 = b1 + 1 #来边的索引 初始=1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2  a2和边b1关联，点都关联上以自己为终点的边（第几条边）
                self.b2a.append(a1) #边关联上起点（添加上是终点的点）
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)#添加上反向边序号
                self.b2revb.append(b1)
                self.n_bonds += 2 #一次性添加2条边，相当于，一个键当做2个有向边处理 
        """

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    
    BatchMolGraph表示一批分子的图形结构和特征。

     BatchMolGraph包含MolGraph的属性以及：
     -smiles_batch：微笑字符串列表。
     -n_mols：批次中的分子数。
     -atom_fdim：原子特征的维数。
     -bond_fdim：键特征的维数（技术上是原子/键特征的组合）。
     -a_scope：元组列表，指示每个分子的起始和终止原子索引。
     b_scope：元组列表，指示每个分子的起始键和终止键索引。
     -max_num_bonds：此批次中与原子相邻的最大键数。
     -b2b ：（可选）从债券索引到传入债券索引的映射。
     -a2a ：（可选）：从原子索引到相邻原子索引的映射。
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        #self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        #atom_messages=1 用点信息 边维数=边初始维数；atom_messages=0用边信息，边维数=边初始维数+点维数

        # Start n_atoms and n_bonds at 1 b/c zero padding 在1 b / c零填充处开始n_atoms和n_bonds
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        #self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule 指示每个分子的（start_atom_index，num_atoms）元组列表
        #self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros 全部以零填充开头，因此以零填充进行索引将返回零
        f_atoms = [[0] * self.atom_fdim]  # atom features  [[0,0,0,...,0 atom_fdim个 0]]
        #f_atoms_gat = [[0] * self.atom_fdim]
        #f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        #a2b = [[]]  # mapping from atom index to incoming bond indices 从原子索引到传入键索引的映射  一个点可能有多个边，所以是列表的列表
        #b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from 从键索引到键来自的原子的索引的映射
        #b2revb = [0]  # mapping from bond index to the index of the reverse bond 从键索引到反向键索引的映射
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)#list.extend 列表末端一次性添加一个列表 MPN
            #f_bonds.extend(mol_graph.f_bonds)
            """
            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
            """
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            #self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            #self.n_bonds += mol_graph.n_bonds

        #self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms) #MPN
        #self.f_atoms_gat = torch.Tensor(f_atoms_gat) #GAT
        """
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        """

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms,self.a_scope
        #return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope 

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    
    将SMILES字符串列表转换为包含分子图批的BatchMolGraph。

     ：param smiles_batch：SMILES字符串的列表。
     ：param args：参数。
     ：return：包含分子的组合分子图的BatchMolGraph
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:#SMILES_TO_GRAPH 已转换的记忆
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)#得到一个列表，是一个分子的信息，包括了初始的点边信息
            if not args.no_cache:#no_cache=False 以下会执行  执行将SMILES_TO_GRAPH里没有的存进去
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)
    
    return BatchMolGraph(mol_graphs, args)
