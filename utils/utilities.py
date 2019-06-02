import numpy as np
import os
import gzip
from scipy import signal
import _pickle as cPickle
import matplotlib.pyplot as plt


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def load_mnist_data():
    dataset = 'mnist.pkl.gz'
    if not os.path.isfile(dataset):
        from six.moves import urllib
        print('downloading data ... (16.2 Mb)')
        urllib.request.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset)
        
    f = gzip.open( dataset,'rb')
    train_set, validate_set, test_set = cPickle.load(f, encoding='latin1')
    [train_x, train_y] = train_set
    [validate_x, validate_y] = validate_set
    [test_x, test_y] = test_set
    f.close()
    
    dataset = {
        'train_x': train_x, 'train_y': train_y, 
        'validate_x': validate_x, 'validate_y': validate_y, 
        'test_x': test_x, 'test_y': test_y}
    
    return dataset


def pytorch_mse(output, target):
    return ((output - target) ** 2).mean(-1)


def pytorch_half_mse(output, target):
    output4d = output.reshape((output.shape[0], 1, 28, 28))
    target4d = target.reshape((target.shape[0], 1, 28, 28))
    return ((output4d[:, :, 0 : 14, :] - target4d[:, :, 0 : 14, :]) ** 2).sum(-1).sum(-1).sum(-1)


def mse(output, target, elementwise=False):
    if elementwise:
        mse_array = []
        for n in range(len(output)):
            mse_array.append(mse(output[n], target[n], elementwise=False))
        return np.array(mse_array)
    else:
        return np.mean(np.square(output.reshape(-1) - target.reshape(-1)))
    

def mse_with_permutation(output_list, target_list, elementwise=False):
    sources_num = len(output_list)
    samples_num = len(output_list[0])
    assert sources_num == 2, 'mse_with_permutation function only support sources_num=2 now!'

    mse_array = []
    order_array = []

    for n in range(samples_num):
        mse_matrix = np.zeros((sources_num, sources_num))
        for k1 in range(sources_num):
            for k2 in range(sources_num):
                mse_matrix[k1, k2] = mse(output_list[k1][n], target_list[k2][n])

        if (mse_matrix[0, 0] + mse_matrix[1, 1]) < (mse_matrix[0, 1] + mse_matrix[1, 0]):
            mse_array.append(mse_matrix[0, 0] + mse_matrix[1, 1])
            order_array.append([0, 1])
        else:
            mse_array.append(mse_matrix[0, 1] + mse_matrix[1, 0])
            order_array.append([1, 0])

    mse_array = np.array(mse_array)
    order_array = np.array(order_array)

    if elementwise:
        return mse_array, order_array
    else:
        return np.mean(mse_array)
            
        
def mse_to_psnr(mse_loss, max):
    return 10. * np.log10(max ** 2 / mse_loss)

    
def psnr(output, target, max, compute_permutation=False):
    if compute_permutation:
        (mse_loss, idx_mat) = mse(output, target, compute_permutation)
    else:
        mse_loss = mse(output, target, compute_permutation)
    psnr_loss = 10. * np.log10(max ** 2 / mse_loss)
    return psnr_loss
    

def plot_image(x, s, s_hat, figure_path):
    """x: (sample_num, 784)"""
    fig, axs = plt.subplots(6, 3, figsize=(4, 8))

    for n in range(6):
        axs[n, 0].imshow(x[n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 1].imshow(s[n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 2].imshow(s_hat[n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)

        for j in range(3):
            axs[n, j].get_xaxis().set_visible(False)
            axs[n, j].get_yaxis().set_visible(False)

    fontsize = 24
    axs[0, 0].set_title(r'$x$', fontsize=fontsize)
    axs[0, 1].set_title(r'$s$', fontsize=fontsize)
    axs[0, 2].set_title(r'$\hat{s}$', fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(figure_path)
    print('Plot figure to {}'.format(figure_path))


def plot_separation_deconvolution(x, s_list, alpha_list, s_hat_list, 
    alpha_hat_list, order_array, figure_path):

    kernel_len = alpha_list[0].shape[-1]

    fig, axs = plt.subplots(6, 9, figsize=(12, 8))
    
    for n in range(6):
        axs[n, 0].imshow(x[n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 1].imshow(s_list[0][n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 2].imshow(s_hat_list[order_array[n, 0]][n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 3].imshow(alpha_list[0][n].reshape(kernel_len, kernel_len), interpolation='nearest', cmap='gray_r', vmin=0, vmax=0.5)
        axs[n, 4].imshow(alpha_hat_list[order_array[n, 0]][n].reshape(kernel_len, kernel_len), interpolation='nearest', cmap='gray_r', vmin=0, vmax=0.5)
        axs[n, 5].imshow(s_list[1][n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 6].imshow(s_hat_list[order_array[n, 1]][n].reshape(28, 28), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
        axs[n, 7].imshow(alpha_list[1][n].reshape(kernel_len, kernel_len), interpolation='nearest', cmap='gray_r', vmin=0, vmax=0.5)
        axs[n, 8].imshow(alpha_hat_list[order_array[n, 1]][n].reshape(kernel_len, kernel_len), interpolation='nearest', cmap='gray_r', vmin=0, vmax=0.5)

        for j in range(9):
            axs[n, j].get_xaxis().set_visible(False)
            axs[n, j].get_yaxis().set_visible(False)

    fontsize = 24
    axs[0, 0].set_title(r'$x$', fontsize=fontsize)
    axs[0, 1].set_title(r'$s_{1}$', fontsize=fontsize)
    axs[0, 2].set_title(r'$\hat{s}_{1}$', fontsize=fontsize)
    axs[0, 3].set_title(r'$\alpha_{1}$', fontsize=fontsize)
    axs[0, 4].set_title(r'$\hat{\alpha}_{1}$', fontsize=fontsize)
    axs[0, 5].set_title(r'$s_{2}$', fontsize=fontsize)
    axs[0, 6].set_title(r'$\hat{s}_{2}$', fontsize=fontsize)
    axs[0, 7].set_title(r'$\alpha_{2}$', fontsize=fontsize)
    axs[0, 8].set_title(r'$\hat{\alpha}_{2}$', fontsize=fontsize)

    plt.savefig(figure_path)
    print('Plot figure to {}'.format(figure_path))


def generate_bar_func(img_size, k, y_bias):
    e = np.zeros((img_size, img_size))
    x_intercept = img_size // 2
    y_intercept = img_size // 2
    
    x = np.arange(-x_intercept, x_intercept).astype(np.float32)
    y = k * x
    
    for i1 in range(-x_intercept, x_intercept):
        e[i1 + x_intercept, int(round(y[i1+x_intercept])) + y_intercept + y_bias] = 1
        
    return e.T
    

def add_gaussian_noise(x, scale, random_state):
    noise_x = x + random_state.uniform(- scale, scale, x.shape)
    return noise_x


def generate_bars_func(img_size, random_state):
    
    k = random_state.uniform(-0.4, 0.4)
    
    e1 = generate_bar_func(img_size, k=k, y_bias=-3)
    e2 = generate_bar_func(img_size, k=k, y_bias=0)
    e3 = generate_bar_func(img_size, k=k, y_bias=3)
    e = e1 + e2 + e3
    
    tmp = random_state.uniform(0., 1.)
    if tmp < 0.5:
        return e
    else:
        return e.T
    

def add_bars(x, random_state):
    x = x.reshape((x.shape[0], 1, 28, 28))
    noise_x = []
    
    for n in range(len(x)):
        bars_img = generate_bars_func(img_size=28, random_state=random_state)
        noise_x.append(x[n, 0] + bars_img)

    noise_x = np.array(noise_x)
    noise_x = noise_x.reshape((noise_x.shape[0], 1, noise_x.shape[1], noise_x.shape[2]))
    noise_x = noise_x.reshape(noise_x.shape[0], 784)
    return noise_x
    
    
def cut_half_image(x):
    x = x.reshape((x.shape[0], 1, 28, 28))
    img_size = x.shape[-1]
    noise_x = np.array(x)
    noise_x[:, :, img_size // 2 :, :] = 0
    noise_x = noise_x.reshape(noise_x.shape[0], 784)
    return noise_x
    
    
def convolve(x, kernel):
    x = x.reshape((x.shape[0], 1, 28, 28))
    
    x_new = np.zeros_like(x)
    for n in range(len(x)):
        x_new[n, 0] = signal.convolve2d(x[n, 0], kernel, mode='same')
    
    x_new = x_new.reshape(x_new.shape[0], 784)
    return x_new
    
    
def random_kernels(size, random_state):
    # assert kernel_size == 5
    kernels = np.array([
               np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]) / 5., 
               np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]) / 5., 
               np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]) / 5., 
               np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 5., 
               np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]) / 15., 
               np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]) / 15., 
               np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]) / 15., 
               np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]) / 15., 
               np.ones((5, 5)) / 25.
              ])
    
    if False:
        fig, axs = plt.subplots(3, 3, sharex=True)
        for i1 in range(3):
            for i2 in range(3):
                axs[i1, i2].imshow(kernels[i1*3+i2], interpolation='nearest', cmap='gray_r', vmin=0., vmax=0.2)
        plt.show()
        pause
    
    idxes = random_state.randint(low=0, high=len(kernels), size=size)
    return kernels[idxes]
    
    
def convolve_each_element(x, kernel):
    x = x.reshape((x.shape[0], 1, 28, 28))
    
    x_new = np.zeros_like(x)
    for n in range(len(x)):
        x_new[n, 0] = signal.convolve2d(x[n, 0], kernel[n], mode='same')
    
    x_new = x_new.reshape(x_new.shape[0], 784)
    return x_new
    
    
def mix_sources(x, mix_num, alphas, random_state, return_individuals=False):
    assert mix_num == len(alphas)
    x = x.reshape((x.shape[0], 1, 28, 28))
    
    indexes = np.arange(len(x))
    
    individuals = []
    mixture = 0.
    
    for k in range(mix_num):
        mixture += alphas[k] * x[indexes]
        individuals.append(x[indexes])
        random_state.shuffle(indexes)
        
    individuals = np.array([e.reshape(e.shape[0], 784) for e in individuals]).transpose(1, 0, 2)  # (samples_num, mix_num, 784)
    mixture = mixture.reshape(mixture.shape[0], 784)

    if return_individuals:
        return mixture, individuals
    else:
        return mixture
        
        
def mix_convolve_sources(x, mix_num, kernels, random_state, return_individuals=False):
    assert mix_num == len(kernels)
    
    indexes = np.arange(len(x))
    
    individuals = []
    mixture = 0.
    
    for k in range(mix_num):
        mixture += convolve(x[indexes], kernels[k])
        individuals.append(x[indexes])
        random_state.shuffle(indexes)
        
    individuals = np.array([e.reshape(e.shape[0], 784) for e in individuals]).transpose(1, 0, 2)  # (samples_num, mix_num, 784)
    mixture = mixture.reshape(mixture.shape[0], 784)

    if return_individuals:
        return mixture, individuals
    else:
        return mixture

def mix_convolve_sources_each_element(x, mix_num, random_state):
    
    indexes = np.arange(len(x))
    
    source_list = []
    kernel_list = []
    mixture = 0.
    
    for k in range(mix_num):
        kernel = random_kernels(size=len(x), random_state=random_state)
        mixture += convolve_each_element(x[indexes], kernel)
        
        source_list.append(x[indexes])
        kernel_list.append(kernel)
        random_state.shuffle(indexes)
        
    mixture = mixture.reshape(mixture.shape[0], 784)

    return mixture, source_list, kernel_list