import numpy as np
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from utilities import create_folder, get_filename, load_mnist_data


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    
    # std = math.sqrt(2. / n)
    std = 0.02
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    # layer.bias.data.copy_(torch.zeros(n_out)) 
    

def init_bn(bn):
    bn.weight.data.fill_(1.)


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        
        nz = 100

        # output size: 512 x 4 x 4
        self.convt1 = nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        # output size: 256 x 8 x 8
        self.convt2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        # output size. 128 x 16 x 16
        self.convt3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        # output size. 1 x 28 x 28
        self.convt4 = nn.ConvTranspose2d(128, 1, 4, 2, 3, bias=False)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.convt1)
        init_layer(self.convt2)
        init_layer(self.convt3)
        init_layer(self.convt4)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        
    def forward(self, input):
        '''input: (batch_size, seed_num)
        '''

        (batch_size, seed_num) = input.shape
        x = input.view(batch_size, seed_num, 1, 1)

        x = F.relu(self.bn1(self.convt1(x)))
        x = F.relu(self.bn2(self.convt2(x)))
        x = F.relu(self.bn3(self.convt3(x)))
        x = torch.sigmoid(self.convt4(x))
        
        return x
        
        
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 128, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.fc_final = nn.Linear(512 * 3 * 3, 1)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc_final)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input):
        '''input: (batch_size, 784)
        '''

        batch_size = input.shape[0]
        x = input.view(batch_size, 1, 28, 28)

        x = F.leaky_relu_(self.conv1(x), 0.2)
        x = F.leaky_relu_(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.bn3(self.conv3(x)), 0.2)
    
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc_final(x)
        x = torch.sigmoid(x)
    
        return x
        

class MNIST(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return len(self.x)

    
def save_result_figure(netG, fig_path, cuda):
    
    z_ = move_data_to_gpu(torch.randn((5 * 6, 100)), cuda)
    
    netG.eval()
    test_images = netG(z_).data.cpu().numpy()
    
    fig, axs = plt.subplots(5, 6, sharex=True)
    for i1 in range(5):
        for j1 in range(6):
            axs[i1, j1].imshow(test_images[(i1*5)+j1, 0], 
                interpolation='nearest', cmap='gray_r')
    plt.savefig(fig_path)
    print("Write out fig to {}".format(fig_path))
    
    
def train():
    
    # Arguments & parameters
    workspace = args.workspace
    filename = args.filename
    cuda = args.cuda and torch.cuda.is_available()

    if cuda:
        print('Using GPU')
    else:
        print('Using CPU')
        
    checkpoints_dir = os.path.join(workspace, filename, 'checkpoints')
    create_folder(checkpoints_dir)
    
    figures_dir = os.path.join(workspace, filename, 'figures')
    create_folder(figures_dir)
    
    # Load data
    dataset = load_mnist_data()    
    
    # Data loader
    train_set = MNIST(dataset['train_x'], dataset['train_y'])
    test_set = MNIST(dataset['test_x'], dataset['test_y'])

    batch_size = 128
    tr_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=True, num_workers=1, pin_memory=True)
    
    # Model
    netG = _netG()
    netD = _netD()

    if cuda:
        netG.cuda()
        netD.cuda()
    
    # Optimizer
    lr = 0.0002
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    iteration = 0

    while(True):

        for (batch_x, batch_y) in tr_loader:
            
            # Save generated figures
            if iteration % 1000 == 0:
                save_result_figure(netG, os.path.join(figures_dir, 
                    '{}_iterations.png'.format(iteration)), cuda)
            
            # Save model
            if iteration % 1000 == 0:
                save_out_dict = {'iteration': iteration, 
                                'netG': netG.state_dict(), 
                                'netD': netD.state_dict() }
                checkpoint_path = os.path.join(checkpoints_dir, 
                    '{}_iterations.pth'.format(iteration))
                torch.save(save_out_dict, checkpoint_path)
                print('Save checkpoint to {}'.format(checkpoint_path))
            
            if batch_x.shape[0] < batch_size:
                break
            
            y_real_ = move_data_to_gpu(torch.ones(batch_size), cuda)
            y_fake_ = move_data_to_gpu(torch.zeros(batch_size), cuda)
            
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)
            
            # Real samples
            netD.train()
            optimizerD.zero_grad()
            D_result = netD(batch_x)
            D_real_loss = F.binary_cross_entropy(D_result, y_real_)

            # Fake samples
            z_ = move_data_to_gpu(torch.randn((batch_size, 100)), cuda)

            G_result = netG(z_)
            D_result = netD(G_result)
            D_fake_loss = F.binary_cross_entropy(D_result, y_fake_)
            
            # Combination loss
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            optimizerD.step()
            
            # Train netG
            netG.train()
            optimizerG.zero_grad()

            z_ = move_data_to_gpu(torch.randn((batch_size, 100)), cuda)
            
            G_result = netG(z_)
            D_result = netD(G_result)
            G_train_loss = F.binary_cross_entropy(D_result, y_real_)
            G_train_loss.backward()
            optimizerG.step()
            
            if iteration % 10 == 0:
                print(iteration, D_train_loss.data.cpu().numpy())
                
            iteration += 1
            
            if iteration == 10001:
                return

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)
    
    if args.mode == "train":
        train()
    else:
        raise Exception("Error!")