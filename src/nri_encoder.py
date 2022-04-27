import torch
import numpy as np
from torch import nn
from collections import OrderedDict


class MLPBlock(nn.Module):
    """The building block for the MLP-based encoder

    """

    def __init__(self, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()

        self.layer1 = nn.Linear(in_dims, hidden_dims)
        self.elu1 = nn.ELU()

        # During training, randomly zeroes some of the elements of the input tensor with probability p
        self.drop_out = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_dims, out_dims)
        self.elu2 = nn.ELU()

        # BatchNormalization along the output dimension
        self.bn = nn.BatchNorm1d(out_dims)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        # reshape back
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, input_batch):
        # Reshape input. The third dimension is #node_dims * #timesteps
        input_batch = input_batch.view(input_batch.size(0),
                                       input_batch.size(1), -1)
        out = self.layer1(input_batch)
        out = self.elu1(out)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.elu2(out)
        return self.batch_norm(out)


class MLPEncoder(nn.Module):

    def __init__(self, in_dims, hidden_dims, out_dims, drop_rate):
        super().__init__()
        self.mlp_block_1 = MLPBlock(in_dims, hidden_dims, hidden_dims,
                                    drop_rate)
        self.mlp_block_2 = MLPBlock(hidden_dims * 2, hidden_dims, hidden_dims,
                                    drop_rate)
        self.out_layer = nn.Linear(hidden_dims, out_dims)
        self.init_weights()

    def init_weights(self):
        """Function for weight initialization
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.1)

    def node2edge(self, input_batch, send_mask, rec_mask):
        """Transferring node representations to edge (sender -> receiver) representations

        Args:
            node_input:
            send_mask: Mask for sender
            rec_mask: Mask for receiver

        Returns:
            Edge representations.
            In the default argument settings, the last 2 dimensions of the tensor are 20 and 2.
        """
        send_batch = torch.matmul(send_mask, input_batch)
        rec_batch = torch.matmul(rec_mask, input_batch)
        return torch.cat([send_batch, rec_batch], dim=2)

    def forward(self, input_batch, send_mask, rec_mask):

        processing_batch = self.mlp_block_1(input_batch)
        # Transfer node to edge. Every edge has a embedded representation (dim=256)
        # `processing_batch` with shape: [#sims, #nodes * (#nodes - 1), hidden_dims * 2]
        processing_batch = self.node2edge(processing_batch, send_mask,
                                          rec_mask)
        # `processing_batch` with shape: [#sims, #nodes * (#nodes - 1), hidden_dims]
        processing_batch = self.mlp_block_2(processing_batch)
        # `output` with shape: [#sims, #nodes * (#nodes - 1), 2]
        output = self.out_layer(processing_batch)

        return output
