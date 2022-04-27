import torch
from torch import nn


class MLPDecoder(nn.Module):
    """Class for MLP-based decoder.
    The decoder takes x_t and as edge types as inputs and output the prediction for the next time step.

    """

    def __init__(self,
                 node_dims,
                 sep_hidden_dims,
                 sep_out_dims,
                 edge_types,
                 hidden_dims,
                 dropout_rate,
                 skip_first=False):
        super().__init__()

        # Construct network strucutures.
        self.layer_sep_1 = nn.ModuleList([
            nn.Linear(2 * node_dims, sep_hidden_dims)
            for i in range(edge_types)
        ])
        self.relu1 = nn.ReLU()
        self.drop_out = nn.Dropout(dropout_rate)
        self.layer_sep_2 = nn.ModuleList([
            nn.Linear(sep_hidden_dims, sep_out_dims) for i in range(edge_types)
        ])
        self.relu2 = nn.ReLU()
        self.sep_out_dims = sep_out_dims
        self.skip_first_edge_type = skip_first

        # The fused layers take both time series (from node) and edge information as inputs.
        self.fussion_layer1 = nn.Linear(node_dims + sep_out_dims, hidden_dims)
        self.fussion_layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.fussion_layer3 = nn.Linear(hidden_dims, node_dims)

    def node2edge(self, node_input, send_mask, rec_mask):
        """Transferring node representations to edge (sender -> receiver) representations

        Args:
            node_input:
            send_mask: Mask for sender
            rec_mask: Mask for receiver

        Returns:
            Edge representations.
            In the default argument settings, the last 2 dimensions of the tensor are 20 and 2.
        """
        send_batch = torch.matmul(send_mask, node_input)
        rec_batch = torch.matmul(rec_mask, node_input)
        return torch.cat([send_batch, rec_batch], dim=-1)

    def one_step_ahead(self, node_input, send_mask, rec_mask,
                       encoder_edge_input):

        # node_input has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # encoder_edge_input has shape:
        # [batch_size, #time_steps, #nodes*(#nodes-1), edge_types]
        edge_rep = self.node2edge(node_input, send_mask, rec_mask)
        fused_input = torch.zeros(edge_rep.size(0), edge_rep.size(1),
                                  edge_rep.size(2), self.sep_out_dims)
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, len(self.layer_sep_1)):
            processed_node_input = self.relu1(self.layer_sep_1[i](edge_rep))
            processed_node_input = self.drop_out(processed_node_input)
            processed_node_input = self.relu2(
                self.layer_sep_2[i](processed_node_input))
            processed_node_input = processed_node_input * encoder_edge_input[:, :, :,
                                                                             i:
                                                                             i +
                                                                             1]
            fused_input += processed_node_input
        # Edge2node
        agg_input = fused_input.transpose(-2, -1).matmul(rec_mask).transpose(
            -2, -1)
        # Skip connection. Fussing x_t (node_input)
        aug_inputs = torch.cat([node_input, agg_input], dim=-1)

        # Output MLP
        pred = self.drop_out(self.relu1(self.fussion_layer1(aug_inputs)))
        pred = self.drop_out(self.relu2(self.fussion_layer2(pred)))
        pred = self.fussion_layer3(pred)

        return node_input + pred

    # Use open source code from https://github.com/ethanfetaya/NRI
    # Copyright (c) 2018 Ethan Fetaya, Thomas Kipf
    def forward(self, inputs, rel_type, send_mask, rec_mask, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [
            rel_type.size(0),
            inputs.size(1),
            rel_type.size(1),
            rel_type.size(2)
        ]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.one_step_ahead(last_pred, rec_mask, send_mask,
                                            curr_rel_type)
            preds.append(last_pred)

        sizes = [
            preds[0].size(0), preds[0].size(1) * pred_steps, preds[0].size(2),
            preds[0].size(3)
        ]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()