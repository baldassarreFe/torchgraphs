from collections import OrderedDict

import torch

from torchgraphs import GraphBatch
from torchgraphs.nn import \
    NodeLinear, EdgeLinear, GlobalLinear, \
    EdgeReLU, NodeReLU, GlobalReLU, \
    EdgeDropout, NodeDropout, GlobalDropout, \
    EdgeBatchNorm, NodeBatchNorm, GlobalBatchNorm
from features_shapes import linear_features
from torchgraphs.data.features import add_random_features


def test_linear_graph_network(graphbatch: GraphBatch, device):
    graphbatch = add_random_features(graphbatch, **linear_features).to(device)

    node_linear: NodeLinear = NodeLinear(
        out_features=linear_features['node_features_shape'],
        incoming_features=linear_features['edge_features_shape'],
        node_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape'],
        aggregation='mean'
    )
    edge_linear: EdgeLinear = EdgeLinear(
        out_features=linear_features['edge_features_shape'],
        edge_features=linear_features['edge_features_shape'],
        sender_features=linear_features['node_features_shape'],
        receiver_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape']
    )
    global_linear: GlobalLinear = GlobalLinear(
        out_features=linear_features['global_features_shape'],
        edge_features=linear_features['edge_features_shape'],
        node_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape'],
        aggregation='mean'
    )

    net = torch.nn.Sequential(OrderedDict([
        ('edge_linear', edge_linear),
        ('edge_dropout', EdgeDropout(p=.2)),
        ('edge_bn', EdgeBatchNorm(num_features=edge_linear.out_features)),
        ('edge_relu', EdgeReLU()),
        
        ('node_linear', node_linear),
        ('node_dropout', NodeDropout(p=.3)),
        ('node_bn', NodeBatchNorm(num_features=node_linear.out_features)),
        ('node_relu', NodeReLU()),
        
        ('global_linear', global_linear),
        ('global_dropout', GlobalDropout(p=.4)),
        ('global_bn', GlobalBatchNorm(num_features=global_linear.out_features)),
        ('global_relu', GlobalReLU()),
    ]))
    net.to(device)

    result: GraphBatch = net(graphbatch)
    result.validate()

    assert graphbatch.num_graphs == result.num_graphs
    assert graphbatch.num_nodes == result.num_nodes
    assert graphbatch.num_edges == result.num_edges

    # These tensors should be passed as they are through the network, never copied
    assert graphbatch.node_index_by_graph is result.node_index_by_graph
    assert graphbatch.edge_index_by_graph is result.edge_index_by_graph
    assert graphbatch.num_nodes_by_graph is result.num_nodes_by_graph
    assert graphbatch.num_edges_by_graph is result.num_edges_by_graph
    assert graphbatch.senders is result.senders
    assert graphbatch.receivers is result.receivers
