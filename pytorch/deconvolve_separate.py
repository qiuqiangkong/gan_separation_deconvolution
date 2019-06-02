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
    pytorch_mse, mse_with_permutation, mse_to_psnr, 
    plot_separation_deconvolution, mix_convolve_sources_each_element)
from pytorch_utils import samplewise_convolve
    

class OptimizerOnInput(object):
    def __init__(self, netG, x, s_list, alpha_list, loss_func, 
        learning_rate, figures_dir):
        '''Optimizer on input z and filter alpha. 

        Inputs:
          netG: pretrained DCGAN model
          x: mixture, (samples_num, 784)
          s_list: list of sources, [(samples_num, 784), (samples_num, 784), ...]
          alpha_list: list of filters, [(samples_num, filter_len, filter_len), 
              (samples_num, filter_len, filter_len), ...]
          loss_func: function
          learning_rate: float
          figures_dir: string
        '''

        self.netG = netG
        self.x = x
        self.s_list = s_list
        self.alpha_list = alpha_list
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.figures_dir = figures_dir

        self.sources_num = len(s_list)
        self.kernel_len = alpha_list[0].shape[-1]
        self.regularize_z = True
        self.seed_num = 100

    def optimize_first_stage(self, repeats_num, max_iteration):
        '''Stage 1: Set several initialization and select the best 
        initialization. 

        Inputs:
          repeats_num: int, number of initializations
          max_iteration: int

        Returns:
          z_hat_list: list of estimated seeds, [(samples_num, seed_num), 
              (samples_num, seed_num), ...]
          alpha_hat_list: list of estimated filters, [(samples_num, filter_len, 
              filter_len), (samples_num, filter_len, filter_len), ...]
          s_hat_list: list of estimated sources, [(samples_num, 1, 28, 28), 
              (samples_num, 1, 28, 28), ...]
          x_hat: list of estimated mixture, (samples_num, 1, 28, 28)
        '''

        # Paths
        first_stage_figures_dir = os.path.join(self.figures_dir, 'first_stage')
        create_folder(first_stage_figures_dir)

        # Repeat mixture and target for applying several initializations
        repeated_x = np.repeat(self.x, repeats=repeats_num, axis=0)

        repeated_s_list = [np.repeat(s, repeats=repeats_num, axis=0) 
            for s in self.s_list]

        repeated_alpha_list = [np.repeat(alpha, repeats=repeats_num, axis=0) 
            for alpha in self.alpha_list]

        samples_num = repeated_x.shape[0]

        # Initialize seed and filter
        z_hat_list = [np.random.normal(loc=0., scale=1, size=(samples_num, 
            self.seed_num)) for _ in range(self.sources_num)]

        alpha_hat_list = [np.random.normal(loc=0., scale=0.1, 
            size=(samples_num, self.kernel_len, self.kernel_len)) 
            for _ in range(self.sources_num)]

        # Optimize on seed and filter
        (z_hat_list, alpha_hat_list, s_hat_list, x_hat) = self.optimize(
            repeated_x, repeated_s_list, repeated_alpha_list, z_hat_list, 
            alpha_hat_list, max_iteration, first_stage_figures_dir)

        # Find the indice of the best initialization for each input mixture
        indices = self.find_best_initialize_indice(x_hat, 
            repeated_x, repeats_num)

        for n in range(len(indices)):
            indices[n] = indices[n] + n * repeats_num

        z_hat_list = [z_hat[indices] for z_hat in z_hat_list]
        alpha_hat_list = [alpha_hat[indices] for alpha_hat in alpha_hat_list]
        s_hat_list = [s_hat[indices] for s_hat in s_hat_list]
        x_hat = x_hat[indices]

        return z_hat_list, alpha_hat_list, s_hat_list, x_hat

    def optimize_second_stage(self, z_hat_list, alpha_hat_list, max_iteration):
        '''Stage 2: Use the initalization obtained from stage 1 and do the 
        optimization on source and filter
        
        Inputs:
          z_hat_list: list of estimated seeds, [(samples_num, seed_num), 
              (samples_num, seed_num), ...]
          alpha_hat_list: list of estimated filters, [(samples_num, filter_len, 
              filter_len), (samples_num, filter_len, filter_len), ...]
          max_iteration: int

        Returns:
          z_hat_list: list of estimated seeds, [(samples_num, seed_num), 
              (samples_num, seed_num), ...]
          alpha_hat_list: list of estimated filters, [(samples_num, filter_len, 
              filter_len), (samples_num, filter_len, filter_len), ...]
          s_hat_list: list of estimated sources, [(samples_num, 1, 28, 28), 
              (samples_num, 1, 28, 28), ...]
          x_hat: list of estimated mixture, (samples_num, 1, 28, 28)
        '''

        second_stage_figures_dir = os.path.join(self.figures_dir, 'second_stage')
        create_folder(second_stage_figures_dir)

        # Optimize
        (z_hat_list, alpha_hat_list, s_hat_list, x_hat) = self.optimize(
            self.x, self.s_list, self.alpha_list, z_hat_list, alpha_hat_list, 
            max_iteration, second_stage_figures_dir)

        return z_hat_list, alpha_hat_list, s_hat_list, x_hat

    def optimize(self, x, s_list, alpha_list, z_hat_list, alpha_hat_list, 
        max_iteration, figures_dir):
        '''Optimize on seed and filter. 

        Input:
          x: mixture, (samples_num, 784)
          s_list: list of ground truth sources, [(samples_num, 784), 
              (samples_num, 784), ...]
          alpha_list: list of ground truth filters, [(samples_num, filter_len, 
              filter_len), (samples_num, filter_len, filter_len), ...]
          z_hat_list: list of estimated seed: [(samples_num, seed_num), 
              (samples_num, seed_num), ...]
          alpha_hat_list: list of estimated filter: [(samples_num, kernel_len, 
              kernel_len), (samples_num, kernel_len, kernel_len), ...]
          max_iteration: int
          figures_dir: string

        Returns:
          z_hat_list: list of estimated seeds, [(samples_num, seed_num), 
              (samples_num, seed_num), ...]
          alpha_hat_list: list of estimated filters, [(samples_num, filter_len, 
              filter_len), (samples_num, filter_len, filter_len), ...]
          s_hat_list: list of estimated sources, [(samples_num, 1, 28, 28), 
              (samples_num, 1, 28, 28), ...]
          x_hat: list of estimated mixture, (samples_num, 1, 28, 28)
        '''

        samples_num = x.shape[0]

        # Estimated seed
        z_hat_list = [Variable(torch.Tensor(z_hat).cuda(), requires_grad=True) 
            for z_hat in z_hat_list]

        # Estimated filter
        alpha_hat_list = [Variable(torch.Tensor(alpha_hat).cuda(), 
            requires_grad=True) for alpha_hat in alpha_hat_list]

        # Mixture
        x = torch.Tensor(x).cuda()

        # Optimizer
        optimizer = optim.Adam(z_hat_list + alpha_hat_list, 
            lr=self.learning_rate, betas=(0.9, 0.999))

        iteration = 0

        while(True):
            if iteration == max_iteration:
                break
            
            self.netG.eval()

            # Estimated sources
            s_hat_list = [self.netG(z_hat) for z_hat in z_hat_list]

            # Estimated mixture
            x_hat = sum([samplewise_convolve(s_hat, alpha_hat) for 
                (s_hat, alpha_hat) in zip(s_hat_list, alpha_hat_list)])

            # Evaluate and plot
            if iteration % 200 == 0:
                # Calculate MSE & PSNR
                np_x = x.data.cpu().numpy()

                np_s_hat_list = [s_hat.data.cpu().numpy() 
                    for s_hat in s_hat_list]

                np_alpha_hat_list = [alpha_hat.data.cpu().numpy() 
                    for alpha_hat in alpha_hat_list]

                (elementwise_mse_loss, elementwise_order_array) = \
                    mse_with_permutation(np_s_hat_list, s_list, elementwise=True)

                elementwise_psnr_loss = mse_to_psnr(elementwise_mse_loss, max=1.)
                    
                print('iteration: {}, mse_loss: {:4f}, psnr: {:4f}'.format(
                    iteration, np.mean(elementwise_mse_loss), 
                    np.mean(elementwise_psnr_loss)))

                # Plot            
                figure_path = '{}/{}_iterations.png'.format(
                    figures_dir, iteration)

                plot_separation_deconvolution(np_x, 
                    s_list, alpha_list, np_s_hat_list, np_alpha_hat_list, 
                    elementwise_order_array, figure_path)
            
            # Loss for backpropagation
            loss = self.loss_func(x_hat.view(samples_num, -1), x)
            
            if self.regularize_z:
                for z_hat in z_hat_list:
                    loss += 1e-3 * (z_hat ** 2).mean(-1)
            
            # Element-wise backpropagation
            loss.backward(torch.ones(samples_num).cuda())

            optimizer.step()
            optimizer.zero_grad()

            iteration += 1

        # Convert to numpy type
        z_hat_list = [z_hat.data.cpu().numpy() for z_hat in z_hat_list]

        alpha_hat_list =[alpha_hat.data.cpu().numpy() 
            for alpha_hat in alpha_hat_list]

        s_hat_list = [s_hat.data.cpu().numpy() for s_hat in s_hat_list]

        x_hat = x_hat.data.cpu().numpy()

        return z_hat_list, alpha_hat_list, s_hat_list, x_hat

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

    # Paths
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

    # Create mixture
    if task == 'deconvolve_separate':
        kernel_size = 5
        sources_num = 2
        (x, s_list, alpha_list) = mix_convolve_sources_each_element(
            data, sources_num, random_state)

        loss_func = pytorch_mse

    # Load model
    netG = _netG()    
    
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()

    psnr_list = []
    
    for n in range(len(x) // batch_size):

        print('====== mini batch {} ======'.format(n))

        batch_x = x[n * batch_size : (n + 1) * batch_size]

        batch_s_list = [s[n * batch_size : (n + 1) * batch_size] 
            for s in s_list]

        batch_alpha_list = [alpha[n * batch_size : (n + 1) * batch_size] 
            for alpha in alpha_list]
        
        # Optimizer for source and filter
        optimizer_on_input = OptimizerOnInput(netG, batch_x, 
            batch_s_list, batch_alpha_list, loss_func=loss_func, 
            learning_rate=0.01, figures_dir=figures_dir)

        # Stage 1: Set several initialization and select the best initialization
        (batch_z_hat_list, batch_alpha_hat_list, batch_s_hat_list, batch_x_hat) \
            = optimizer_on_input.optimize_first_stage(
            repeats_num, max_iteration=201)

        # Stage 2: Use the initalization obtained from stage 1 and do the 
        # optimization on source and filter
        (batch_z_hat_list, batch_alpha_hat_list, batch_s_hat_list, batch_x_hat) \
            = optimizer_on_input.optimize_second_stage(
            batch_z_hat_list, batch_alpha_hat_list, max_iteration=5001)

        # Calculate psnr
        (elementwise_mse_loss, _) = mse_with_permutation(batch_s_hat_list, 
            batch_s_list, elementwise=True)

        elementwise_psnr_loss = mse_to_psnr(elementwise_mse_loss, max=1.)

        psnr_loss = np.mean(elementwise_psnr_loss)

        psnr_list.append(psnr_loss)
        print('*** mean psnr on {} mini batches: {}'.format(n, np.mean(psnr_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--task', type=str, choices=['deconvolve_separate'], required=True)
    parser.add_argument('--repeats_num', type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)
    
    main(args)