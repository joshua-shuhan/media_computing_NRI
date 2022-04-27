import time
import torch

import numpy as np
from torch import nn
import argparse
from src.utils import mask, load_data
from src.nri_encoder import MLPEncoder

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    type=str,
    dest='model_path',
    default='saved_model/encoder/20220422_0942/9/',
    help='Specified model path within saved_model/encoder folder')
parser.add_argument('-ds',
                    '--data_suffix',
                    dest='data_suffix',
                    default='_springsLight5')
parser.add_argument('-n', '--nodes', dest='nodes', default=5)
parser.add_argument('-d', '--node_dims', dest='nodes_dims', default=4)
parser.add_argument('-hid', '--hidden', dest='hidden_dims', default=256)
parser.add_argument('-ts', '--time_steps', dest='time_steps',
                    default=49)  
parser.add_argument('-et', '--edge_types', dest='edge_types', default=2)
parser.add_argument('-cuda', '--cuda', dest='cuda', default=True)
parser.add_argument('-dr', '--dropout_rate', dest='dropout', default=0.01)
parser.add_argument('-b', '--batch_size', dest='batch_size', default=5)

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

args = parser.parse_args()
send_mask, rec_mask = mask(args.nodes)

model = MLPEncoder(args.time_steps * args.nodes_dims, args.hidden_dims,
                   args.edge_types, args.dropout)
model.load_state_dict(torch.load(args.model_path + 'model.ckpt'))

if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    model.cuda()
    send_mask.cuda()
    rec_mask.cuda()
if args.cuda and torch.cuda.is_available():
    print('Run in GPU')
else:
    print('No GPU provided.')


train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, suffix=args.data_suffix, root=True)

ret_output = np.zeros([
    len(test_loader), args.batch_size * args.nodes * (args.nodes - 1),
    args.edge_types
])
ret_target = np.zeros(
    [len(test_loader), args.batch_size * args.nodes * (args.nodes - 1)])

for batch_idx, (data, target) in enumerate(test_loader):
    if args.cuda and torch.cuda.is_available():
        data.cuda()
        target.cuda()
    data = data[:, :, :args.time_steps, :]

    output = model(data, send_mask, rec_mask)
    output = output.view(-1, args.edge_types)
    target = target.view(-1)

    ret_output[batch_idx, :, :] = output.detach().numpy()
    ret_target[batch_idx, :] = target

with open(f'saved_results/encoder_result/{args.data_suffix+"output"}.npy',
          'wb') as f:
    np.save(f, ret_output)

with open(f'saved_results/encoder_result/{args.data_suffix+"target"}.npy',
          'wb') as f:
    np.save(f, ret_target)

print('----------------Generated results have been saved-------------------')
