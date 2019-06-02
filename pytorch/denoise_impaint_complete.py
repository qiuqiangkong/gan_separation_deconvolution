import numpy as np
import time
import os
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

import _pickle as cPickle

from main import _netG
from utilities import (get_filename, create_folder, load_mnist_data, 
    add_gaussian_noise, add_bars, cut_half_image, pytorch_mse, 
    pytorch_half_mse, mse, mse_to_psnr, plot_image)
    

class OptimizerOnInput(object):
    def __init__(self, netG, x, s, loss_func, learning_rate, figures_dir):
        '''Optimizer on input z and filter alpha. 

        Inputs:
          netG: pretrained DCGAN model
          x: mixture, (samples_num, 784)
          s: ground truth source, (samples_num, 784)
          loss_func: function
          learning_rate: float
          figures_dir: string
        '''

        self.netG = netG
        self.x = x
        self.s = s
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.figures_dir = figures_dir

        self.regularize_z = True
        self.seed_num = 100

    def optimize_first_stage(self, repeats_num, max_iteration):
        '''Stage 1: Set several initialization and select the best 
        initialization. 

        Inputs:
          repeats_num: int, number of initializations
          max_iteration: int

        Returns:
          z_hat: estimated seed, (samples_num, seed_num)
          alpha_hat: estimated filters, (samples_num, filter_len, filter_len)
          s_hat: estimated source, (samples_num, 1, 28, 28)
          x_hat: estimated mixture
        '''

        # Paths
        first_stage_figures_dir = os.path.join(self.figures_dir, 'first_stage')
        create_folder(first_stage_figures_dir)

        # Repeat mixture and target for applying different initializations for 
        # a single mixture
        repeated_x = np.repeat(self.x, repeats=repeats_num, axis=0)
        repeated_s = np.repeat(self.s, repeats=repeats_num, axis=0)
        samples_num = repeated_x.shape[0]

        # Initialize seed and filter
        z_hat = np.random.normal(loc=0., scale=1, size=(samples_num, 
            self.seed_num))

        alpha_hat = np.ones(samples_num)

        # Optimize on seed and filter
        (z_hat, alpha_hat, s_hat, x_hat) = self.optimize(repeated_x, 
            repeated_s, z_hat, alpha_hat, max_iteration, first_stage_figures_dir)

        # Find the indice of the best initialization for each input mixture
        indices = self.find_best_initialize_indice(x_hat, repeated_x, 
            repeats_num)

        for n in range(len(indices)):
            indices[n] = indices[n] + n * repeats_num

        z_hat = z_hat[indices]
        alpha_hat = alpha_hat[indices]
        s_hat = s_hat[indices]
        x_hat = x_hat[indices]

        return z_hat, alpha_hat, s_hat, x_hat

    def optimize_second_stage(self, z_hat, alpha_hat, max_iteration):
        '''Stage 2: Use the initalization obtained from stage 1 and do the 
        optimization on source and filter

        Inputs:
          z_hat: estimated seed, (samples_num, seed_num)
          alpha_hat: estimated filters, (samples_num, filter_len, filter_len)
          max_iteration: int

        Returns:
          z_hat: estimated seed, (samples_num, seed_num)
          alpha_hat: estimated filters, (samples_num, filter_len, filter_len)
          s_hat: estimated source, (samples_num, 1, 28, 28)
          x_hat: estimated mixture
        '''

        second_stage_figures_dir = os.path.join(self.figures_dir, 'second_stage')
        create_folder(second_stage_figures_dir)

        # Optimize
        (z_hat, alpha_hat, s_hat, x_hat) = self.optimize(self.x, self.s, z_hat, alpha_hat, 
            max_iteration, second_stage_figures_dir)

        return z_hat, alpha_hat, s_hat, x_hat

    def optimize(self, x, s, z_hat, alpha_hat, max_iteration, figures_dir):
        '''Optimize on seed and filter. 

        Input:
          z_hat: estimated seed, (samples_num, seed_num)
          alpha_hat: estimated filters, (samples_num, filter_len, filter_len)
          s_hat: estimated source, (samples_num, 1, 28, 28)
          x_hat: estimated mixture
          max_iteration: int
          figures_dir: string

        Returns:
          z_hat: estimated seed, (samples_num, seed_num)
          alpha_hat: estimated filters, (samples_num, filter_len, filter_len)
          s_hat: estimated source, (samples_num, 1, 28, 28)
          x_hat: estimated mixture
        '''

        samples_num = x.shape[0]

        # Estimated seed
        z_hat = Variable(torch.Tensor(z_hat).cuda(), requires_grad=True)

        # Estimated filter
        alpha_hat = Variable(torch.Tensor(alpha_hat).cuda(), requires_grad=True)

        # Mixture
        x = torch.Tensor(x).cuda()

        # Optimizer
        optimizer = optim.Adam([z_hat, alpha_hat], lr=self.learning_rate, 
            betas=(0.9, 0.999))

        iteration = 0

        while(True):
            if iteration == max_iteration:
                break
            
            self.netG.eval()

            # Estimated source
            s_hat = self.netG(z_hat)

            # Estimated x
            x_hat = alpha_hat[:, None, None, None] * s_hat

            # Evaluate and plot
            if iteration % 200 == 0:
                # Calculate MSE & PSNR
                np_x = x.data.cpu().numpy()
                np_s_hat = s_hat.data.cpu().numpy()

                elementwise_mse_loss = mse(np_s_hat, s, elementwise=True)
                elementwise_psnr_loss = mse_to_psnr(elementwise_mse_loss, max=1.)

                print('iteration: {}, mse_loss: {:4f}, psnr: {:4f}'.format(
                    iteration, np.mean(elementwise_mse_loss), 
                    np.mean(elementwise_psnr_loss)))
            
                # Plot
                figure_path = '{}/{}_iterations.png'.format(figures_dir, iteration)
                plot_image(np_x, s, np_s_hat, figure_path)
                print('Save to png to {}'.format(figure_path))
            
            # Loss for backpropagation
            loss = self.loss_func(x_hat.view(samples_num, -1), x)
            
            if self.regularize_z:
                loss += 1e-3 * (z_hat ** 2).mean(-1)
            
            # Element-wise backpropagation
            loss.backward(torch.ones(samples_num).cuda())

            optimizer.step()
            optimizer.zero_grad()

            iteration += 1

        z_hat = z_hat.data.cpu().numpy()
        alpha_hat = alpha_hat.data.cpu().numpy()
        s_hat = s_hat.data.cpu().numpy()
        x_hat = x_hat.data.cpu().numpy()

        return z_hat, alpha_hat, s_hat, x_hat

    def find_best_initialize_indice(self, repeated_x_hat, repeated_x,
         repeats_num):
        '''Find the indice of the best initialization for each input mixture. 

        Input:
          repeated_x_hat: (samples_num x repeats_num, 1, 28, 28)
          repeated_x: (samples_num x repeats_num, 784)
          repeates_num: int

        Returns:
          indices: (samples_num,)
        '''
        repeated_x_hat_3d = repeated_x_hat.reshape((-1, repeats_num, 784))
        repeated_x_3d = repeated_x.reshape((-1, repeats_num, 784))
        
        mse_2d = np.mean(np.square(repeated_x_hat_3d - repeated_x_3d), axis=-1)
        
        indices = np.argmin(mse_2d, axis=-1)
        
        return indices


def main(args):

    # Arguments & parameters
    workspace = args.workspace
    task = args.task
    repeats_num = args.repeats_num
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    batch_size = 200

    # Path
    if cuda:
        print('Using GPU')
    else:
        print('Using CPU')

    checkpoint = torch.load(os.path.join(workspace, 'main', 'checkpoints', 
        '10000_iterations.pth'))

    figures_dir = os.path.join(workspace, filename, task, 
        'repeats_{}'.format(repeats_num), 'figures')
    create_folder(figures_dir)
    
    random_state = np.random.RandomState(1111)
    
    # Load data
    dataset = load_mnist_data()           
    data = dataset['test_x']

    # Add noise
    if task == 'denoise':
        std = np.std(data)
        x = add_gaussian_noise(data, std, random_state)
        loss_func = pytorch_mse

    elif task == 'impaint':
        x = add_bars(data, random_state)
        loss_func = pytorch_mse

    elif task == 'complete':
        x = cut_half_image(data)
        loss_func = pytorch_half_mse

    # Source signal
    s = data

    # Load model
    netG = _netG()    
    
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()

    psnr_list = []
    
    for n in range(len(x) // batch_size):

        print('====== mini batch {} ======'.format(n))

        batch_x = x[n * batch_size : (n + 1) * batch_size]
        batch_s = s[n * batch_size : (n + 1) * batch_size]
        
        # Optimizer for source and filter
        optimizer_on_input = OptimizerOnInput(netG, batch_x, 
            batch_s, loss_func=loss_func, learning_rate=0.01, 
            figures_dir=figures_dir)

        # Stage 1: Set several initialization and select the best initialization
        (batch_z_hat, batch_alpha_hat, batch_s_hat, batch_x_hat) = \
            optimizer_on_input.optimize_first_stage(repeats_num, 
            max_iteration=201)

        # Stage 2: Use the initalization obtained from stage 1 and do the 
        # optimization on source and filter
        (batch_z_hat, batch_alpha_hat, batch_s_hat, batch_x_hat) = \
            optimizer_on_input.optimize_second_stage(batch_z_hat, 
            batch_alpha_hat, max_iteration=5001)

        # Calculate psnr
        elementwise_mse_loss = mse(batch_s_hat, batch_s, elementwise=True)
        elementwise_psnr_loss = mse_to_psnr(elementwise_mse_loss, max=1.)
        psnr_loss = np.mean(elementwise_psnr_loss)

        psnr_list.append(psnr_loss)
        print('*** mean psnr on {} mini batches: {}'.format(n, np.mean(psnr_list)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--task', type=str, choices=['denoise', 'impaint', 'complete'], required=True)
    parser.add_argument('--repeats_num', type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)
    
    main(args)
        