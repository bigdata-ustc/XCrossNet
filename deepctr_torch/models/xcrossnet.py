# -*- coding:utf-8 -*-
'''
Author:
    Runlong Yu, Zihan Wang
Reference:
    [1] Runlong Yu, Yuyang Ye, Qi Liu, Zihan Wang, Chunfeng Yang, Yucheng Hu, and Enhong Chen. XCrossNet: Feature Structure-Oriented Learning for Click-Through Rate Prediction. In: PAKDD. (2021)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, DenseFeat
from ..layers import DNN, concat_fun, InnerProductLayer, OutterProductLayer


class XCrossNet(BaseModel):
    """Instantiates the Product-based Neural Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5,
                 l2_reg_dnn=0, cross_num=3, use_linear=False, use_cross=False,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation=F.relu, use_inner=True, use_outter=False,
                 kernel_type='mat', task='binary', device='cpu', ):

        super(XCrossNet, self).__init__(dnn_feature_columns if use_linear else [], dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding,
                                   l2_reg_linear=0, init_std=init_std, seed=seed,
                                   task=task, device=device)

        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        self.task = task
        self.cross_num = cross_num
        self.use_linear = use_linear
        self.use_cross = use_cross

        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, embedding_size, include_dense=False,
                                            feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)

        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, embedding_size, kernel_type=kernel_type, device=device)
        self.dense_features = len(list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else [])
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(self.dense_features, 1)), requires_grad=True)
             for i in range(self.cross_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(self.dense_features, 1)), requires_grad=True)
             for i in range(self.cross_num)])

        dnn_input_features = product_out_dim + self.compute_input_dim(dnn_feature_columns, embedding_size)\
                             + self.dense_features * self.cross_num
        if use_cross:
            self.dnn_kernel = nn.Parameter(nn.init.xavier_normal_(torch.empty(dnn_input_features, 1)),
                                           requires_grad=True)
            self.dnn_bias = nn.Parameter(nn.init.zeros_(torch.empty(dnn_input_features, 1)), requires_grad=True)
            dnn_input_features *= 2

        self.dnn = DNN(dnn_input_features, dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)

        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        # product part
        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal
        dnn_input = combined_dnn_input([product_layer], [])

        # cross part
        if len(dense_value_list) > 0:
            cross_input = combined_dnn_input([], dense_value_list)
            x_0 = cross_input.unsqueeze(2)
            cross_output = cross_input
            x_1 = x_0
            for i in range(self.cross_num):
                x_2 = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
                x_2 = torch.matmul(x_0, x_2)
                x_2 = x_2 + self.bias[i]
                cross_output = torch.cat((x_2.squeeze(2), cross_output), dim=-1)
                x_1 = x_2
            dnn_input = torch.cat([cross_output, dnn_input], dim=-1)

        if self.use_cross:
            x_0 = dnn_input.unsqueeze(2)
            x_1 = torch.tensordot(x_0, self.dnn_kernel, dims=([1], [0]))
            x_1 = torch.matmul(x_0, x_1)
            x_1 = x_1 + self.dnn_bias
            dnn_input = torch.cat((x_1.squeeze(2), dnn_input), dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        if self.use_linear:
            logit += self.linear_model(X)

        y_pred = self.out(logit)

        return y_pred