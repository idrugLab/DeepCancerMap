from argparse import Namespace

import torch
import torch.nn as nn

from deepcancer.fpgnn.nn_utils import get_activation_function, initialize_weights
from .mpn import FPN, GAT


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    # “ MoleculeModel是一个模型，其中包含一个消息传递网络，后面是前馈层。
    # 使用pytorch的时候，模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用forward，forward必须重写，可以看成，模型运行的方法。

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.
        初始化函数，每建立一个实例则主动执行一次
        :param classification: Whether the model is a classification model.
        还要传入2个bool值，classification和multiclass
        """
        super(MoleculeModel, self).__init__()
        # Module类的构造函数

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
            # 是分类，那么用sigmoid = nn.Sigmoid() 使用方式如下
            # test = torch.tensor([1, 5, 4, 8, 9])
            # s = nn.Sigmoid()
            # print(s(test))
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        # assert断言，后面结果是not则对，不是则抛出异常

    '''
    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.
        为模型创建消息传递编码器。
        :param args: Arguments.
        """
        self.encoder = MPN(args)
    '''

    def create_scale(self, args: Namespace):

        # 创建GAT FPN比例的Linear

        self.gat_scale = args.gat_scale
        self.gat_dim = (300 * 2 * self.gat_scale) // 1
        self.gat_dim = int(self.gat_dim)
        self.fc_gat = nn.Linear(300, self.gat_dim)
        self.fc_fpn = nn.Linear(300, 600 - self.gat_dim)
        self.act_func = get_activation_function(args.activation)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.
        为模型创建前馈网络。
        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:  # features_only=false
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:  # 运行时 'ffn_num_layers': 2,
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:  # ffn_num_layers=2
            ffn = [
                dropout,
                nn.Linear(in_features=600, out_features=300, bias=True)  # 【600->300】
                # nn.Linear(first_linear_dim, args.ffn_hidden_size)#Linear(in_features=300, out_features=300, bias=True)
            ]
            for _ in range(args.ffn_num_layers - 2):  # ffn_num_layers=2 这个不执行
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),  # Linear(in_features=300, out_features=1, bias=True)
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        # *ffn是把ffn列表里的东西都传入进去
        # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。另外，也可以传入一个有序模块。
        # 使用torch.nn.Module，我们可以根据自己的需求改变传播过程，如RNN等
        # 如果你需要快速构建或者不需要过多的过程，直接使用torch.nn.Sequential即可。
        # Sequential使用实例
        # model = nn.Sequential(
        #          nn.Conv2d(1,20,5),
        #          nn.ReLU(),
        #          nn.Conv2d(20,64,5),
        #          nn.ReLU()
        #        )

        # Sequential with OrderedDict使用实例
        # model = nn.Sequential(OrderedDict([
        #          ('conv1', nn.Conv2d(1,20,5)),
        #          ('relu1', nn.ReLU()),
        #          ('conv2', nn.Conv2d(20,64,5)),
        #          ('relu2', nn.ReLU())
        #        ]))

    def create_fpn(self, args: Namespace):

        # 创建指纹的层  像encoder一样在mpn里写 然后和encoder一样并在一起用

        self.encoder2 = FPN(args)

        # 和MPN放在一个文件里

    def create_gat(self, args: Namespace):
        """
        创建GAT层，GAT+FPN。 GAT层的输出是300维
        """
        self.encoder3 = GAT(args)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.
        在输入上运行MoleculeModel。
        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output1 = self.encoder3(*input)  # GAT的输出
        smi_list = input
        output2 = self.encoder2(smi_list)
        output1 = output1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        output1 = self.fc_gat(output1)
        output1 = self.act_func(output1)
        output2 = output2.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        output2 = self.fc_fpn(output2)
        output2 = self.act_func(output2)
        output3 = torch.cat([output1, output2], axis=1)
        output = self.ffn(output3)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss 在使用BCEWithLogitsLoss进行b / c训练期间不要应用sigmoid
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    """
    构建一个MoleculeModel，它是通过神经网络+前馈层传递的消息。

     ：param args：参数。
     ：return：一个MoleculeModel，包含MPN编码器以及带有已初始化参数的最终线性层。
    """
    # nn.Module是torcn.nn中的类
    output_size = args.num_tasks
    # 最终输出预测的size，多少个task就多大size
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass')
    # 实例化一个MoleculeModel类
    # 实例化后自动执行__init__和forward
    # model.create_encoder(args)
    model.create_gat(args)
    model.create_fpn(args)
    # 编码器
    model.create_scale(args)
    model.create_ffn(args)
    # 前馈神经网络

    initialize_weights(model)
    # 初始化权重

    return model
