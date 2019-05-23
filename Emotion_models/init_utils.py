import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(model):
    for key in model.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(model.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                model.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            model.state_dict()[key][...] = 0
    for m in model.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()