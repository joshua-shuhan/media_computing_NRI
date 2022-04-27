import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from src.nri_decoder import MLPDecoder
from src.utils import mask, load_data

parser = argparse.ArgumentParser()


def parse_args():
    """Parse argumetns

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser(description="code for training decoder")
    parser.add_argument(
        '-n',
        '--node_dims',
        dest='node_dims',
        type=int,
        help="Dimensions of each nodes in the intracting system",
        default=4)
    parser.add_argument(
        '-sd',
        '--sep_hidden_dims',
        dest='sep_hidden_dims',
        type=int,
        help=
        "Dimension of hidden states in the edge-type specific Neural network",
        default=256)
    parser.add_argument(
        '-so',
        '--sep_out_dims',
        dest='sep_out_dims',
        type=int,
        help="Dimension of out in the edge-type specific Neural network",
        default=256)
    parser.add_argument('-hid',
                        '--hidden_dims',
                        dest='hidden_dims',
                        type=int,                        
                        help="Dimension of hidden states in the fused network",
                        default=256)
    parser.add_argument('-e',
                        '--epoch',
                        dest='epoch_num',
                        type=int,                        
                        help="Number of trainning epochs",
                        default=30)
    parser.add_argument('-ps',
                        '--time_step',
                        dest='time_steps_test',
                        type=int,                        
                        help="Number of time steps in the training dataset",
                        default=49)
    parser.add_argument('-pred_s',
                        '--pred_step',
                        dest='pred_steps',
                        type=int,
                        help="Prediction time steps",
                        default=1)
    parser.add_argument('-et',
                        '--edge_types',
                        dest='edge_types',
                        type=int,
                        help="Number of edge types",
                        default=2)
    parser.add_argument('-dr',
                        '--dropout_rate',
                        dest='dropout',
                        type=float,
                        help="Dropout rate",
                        default=0.05)
    parser.add_argument('-nn',
                        '--num_nodes',
                        dest='num_nodes',
                        type=int,
                        help="Number of nodes in the interacting system",
                        default=5)
    parser.add_argument('-c',
                        '--cuda',
                        dest='cuda',
                        help="Use cuda or not",
                        default=False)

    return parser.parse_args()


def reconstruction_error(pred, target):
    """This function computes the error between prediction trajectory and target trajectory.

    Args:
        pred:
        target:

    Returns:
        Mean prediction error.
    """
    loss = ((pred - target)**2)
    return loss.sum() / (loss.size(0) * loss.size(1))


def train(args):
    """Train and validate the decoder model

    Args:
        args: pased argument. See 'def parse_args()' for details

    """

    # Designate number of training epochs, optimizer and scheduler
    epoch_nums = args.epoch_num
    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # timestr will be used in model path.
    timestr = time.strftime("%Y%m%d_%H%M")
    parent_folder = f'../saved_model/decoder/{timestr}'

    # Create a directory if not exist.
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)

    # Just a random preset. minimum_loss will be updated during training.
    minimum_loss = 100
    best_model_path = ''
    for i in range(epoch_nums):
        loss_train = []
        loss_val = []
        # model.train() tells your model that you are training the model
        model.train()
        # The path for trained model, with timestr and epoch index i.
        # If model is improved, the model will be saved to this direcotry.
        model_path = f'../saved_model/decoder/{timestr}/{i}/'
        for batch_index, (input_batch, relations) in enumerate(train_loader):
            # rel_type_onehot is one-hot encoding for the ground truth relations
            rel_type_onehot = torch.FloatTensor(input_batch.size(0),
                                                rec_mask.size(0),
                                                args.edge_types)
            rel_type_onehot.zero_()
            rel_type_onehot.scatter_(
                2, relations.view(input_batch.size(0), -1, 1), 1)

            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                rel_type_onehot.cuda()

            output = model(input_batch, rel_type_onehot, send_mask, rec_mask,
                           args.pred_steps)
            # Decoder performs prediction task. The target is just one time step ahead of input.
            target = input_batch[:, :, 1:, :]

            loss = reconstruction_error(output, target)
            optimizer.zero_grad()
            loss.backward()

            # All optimizers implement a step() method, that updates the parameters.
            # Learning rate scheduling should be applied after optimizer’s update.
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())

        # Validate model performance
        model.eval()
        for batch_index, (input_batch, relations) in enumerate(valid_loader):
            rel_type_onehot = torch.FloatTensor(input_batch.size(0),
                                                rec_mask.size(0),
                                                args.edge_types)
            rel_type_onehot.zero_()
            rel_type_onehot.scatter_(
                2, relations.view(input_batch.size(0), -1, 1), 1)

            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                rel_type_onehot.cuda()

            output = model(input_batch, rel_type_onehot, send_mask, rec_mask,
                           args.pred_steps)
            target = input_batch[:, :, 1:, :]

            loss = reconstruction_error(output, target)

            loss_val.append(loss.item())

        # According to the validation loss, save the model if the model performance is improved.
        # Print out messages
        if np.mean(loss_val) < minimum_loss:
            minimum_loss = np.mean(loss_val)
            best_model_path = model_path
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            torch.save(model.state_dict(), best_model_path + 'model.ckpt')
            pickle.dump({'args': args},
                        open(best_model_path + 'model_args.pkl', "wb"))
            print('-----------------------------------------------')
            print(
                f'epoch {i} decoder training finish. Model performance improved.'
            )
            print(f'validation loss {np.mean(loss_val)}')
            print(f'save best model to {best_model_path}')

    return best_model_path


#------------------------------------------------------------
def test(args, best_model_path):
    # Test the best performance model (selected based on validation loss).
    loss_test = []
    loss_baseline_test = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.load_state_dict(torch.load(best_model_path + 'model.ckpt'))
    for batch_idx, (input_batch, relations) in enumerate(test_loader):
        rel_type_onehot = torch.FloatTensor(input_batch.size(0),
                                            rec_mask.size(0), args.edge_types)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, relations.view(input_batch.size(0), -1, 1),
                                 1)
        input_batch = input_batch[:, :,
                                  -args.time_steps_test:, :]  

        if args.cuda and torch.cuda.is_available():
            input_batch.cuda()
            rel_type_onehot.cuda()
        target = input_batch[:, :, 1:, :]
        output = model(input_batch, rel_type_onehot, send_mask, rec_mask,
                       args.pred_steps)

        loss = reconstruction_error(output, target)
        loss_baseline = reconstruction_error(input_batch[:, :, :-1, :],
                                             input_batch[:, :, 1:, :])

        loss_test.append(loss)
        loss_baseline_test.append(loss_baseline)

    print('-------------testing finish-----------------')
    print(f'load model from: {best_model_path}')
    print(f'test reconstruction error: {sum(loss_test) / len(loss_test)}')
    print(f'test baseline loss: {np.mean(loss_baseline_test)}')


if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)

    # args, model, loaders and masks are global variable
    args = parse_args()
    send_mask, rec_mask = mask(args.num_nodes)
    model = MLPDecoder(args.node_dims, args.sep_hidden_dims, args.sep_out_dims,
                       args.edge_types, args.hidden_dims, args.dropout)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        model.cuda()
        send_mask.cuda()
        rec_mask.cuda()
    if args.cuda and torch.cuda.is_available():
        print('Run in GPU.')
    else:
        print('No GPU provided.')

    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
        batch_size=5, suffix='_springsLight5', root=False)

    best_model_path = train(args)

    test(args, best_model_path)
