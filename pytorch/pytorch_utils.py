import torch
import torch.nn.functional as F


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
    

def samplewise_convolve(s, alpha):
    img_size = 28
    kernel_size = alpha.shape[-1]
    
    def _convolve(_s, _alpha):
        tmp = F.conv2d(_s.view(1, 1, img_size, img_size), 
            _alpha.view(1, 1, kernel_size, kernel_size), 
            padding=(kernel_size-1) // 2)

        return tmp.view(1, img_size, img_size)
    
    return torch.stack([_convolve(_s, _alpha) for (_s, _alpha) in zip(torch.unbind(s, dim=0), torch.unbind(alpha, dim=0))])