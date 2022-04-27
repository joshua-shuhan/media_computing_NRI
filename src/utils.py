import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader



def mask(num_nodes):
    # input_batch: [#sims, #nodes, ...]
    sender_mask = np.zeros([num_nodes-1, num_nodes])
    sender_mask[:,0]=1

    for i in range(num_nodes-1):
        temp = np.zeros([num_nodes-1, num_nodes])
        temp[:, i+1] = 1
        sender_mask = np.concatenate((sender_mask, temp))
        
    receiver_mask = np.eye(num_nodes)
    receiver_mask = np.delete(receiver_mask, (0), axis=0)
    
    for i in range(num_nodes-1):
        temp = np.eye(num_nodes)
        temp = np.delete(temp, (i+1), axis=0)
        receiver_mask = np.concatenate((receiver_mask, temp))
    
    # send_batch = torch.matmul(sender_mask, input_batch)
    # rec_batch = torch.matmul(receiver_mask, input_batch)
    
    return torch.FloatTensor(sender_mask), torch.FloatTensor(receiver_mask)

# def edge2node(input_batch, rec_mask):
#     num_nodes = input_batch.size(1)
#     receiver_mask = np.eye(num_nodes)
#     receiver_mask = np.delete(receiver_mask, (0), axis=0)
    
#     for i in range(num_nodes-1):
#         temp = np.eye(num_nodes)
#         temp = np.delete(temp, (i+1), axis=0)
#         receiver_mask = np.concatenate((receiver_mask, temp))
    
#     send_batch = torch.matmul(sender_mask, input_batch)
#     rec_batch = torch.matmul(receiver_mask, input_batch)


def load_data(batch_size=1, suffix='', root=False):
    if not root:
        loc_train = np.load('../data/loc_train' + suffix + '.npy')
        vel_train = np.load('../data/vel_train' + suffix + '.npy')
        edges_train = np.load('../data/edges_train' + suffix + '.npy')

        loc_valid = np.load('../data/loc_valid' + suffix + '.npy')
        vel_valid = np.load('../data/vel_valid' + suffix + '.npy')
        edges_valid = np.load('../data/edges_valid' + suffix + '.npy')

        loc_test = np.load('../data/loc_test' + suffix + '.npy')
        vel_test = np.load('../data/vel_test' + suffix + '.npy')
        edges_test = np.load('../data/edges_test' + suffix + '.npy')
    else:
        loc_train = np.load('data/loc_train' + suffix + '.npy')
        vel_train = np.load('data/vel_train' + suffix + '.npy')
        edges_train = np.load('data/edges_train' + suffix + '.npy')

        loc_valid = np.load('data/loc_valid' + suffix + '.npy')
        vel_valid = np.load('data/vel_valid' + suffix + '.npy')
        edges_valid = np.load('data/edges_valid' + suffix + '.npy')

        loc_test = np.load('data/loc_test' + suffix + '.npy')
        vel_test = np.load('data/vel_test' + suffix + '.npy')
        edges_test = np.load('data/edges_test' + suffix + '.npy')
        
    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min