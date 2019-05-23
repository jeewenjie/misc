import os
import logging
import torch
import torch.nn as nn
from .vgg import vgg_utils
from .vgg.vgg_gru import vgg_gru
from .vgg.vgg_lstm import vgg_lstm
from .vgg.vgg_bilstm import vgg_bilstm
from .vgg.vgg_bilstm_nonlocal import vgg_bilstm_nonlocal
from .vgg.vgg_gru_nonlocal import vgg_gru_nonlocal
from .resnet import resnet_utils
from .resnet.resnet_gru import resnet_gru
from .resnet.resnet_lstm import resnet_lstm
from .resnet.resnet_bilstm import resnet_bilstm
from .resnet.resnet_bilstm_nonlocal import resnet_bilstm_nonlocal
from .resnet.resnet_gru_nonlocal import resnet_gru_nonlocal
from .c3d.c3d import c3d
from .init_utils import weights_init

class EmotionNet(nn.Module):
    """
    Build Emotion Recognition Model
    """

    def __init__(self, arch, num_classes):
        super(EmotionNet, self).__init__()
        self.arch = arch
        self.model = eval(arch)(num_classes=num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def inital_weights(self):
        logging.info('Initing Emotion Recognition network...')
        self.model.apply(weights_init)
        logging.info('Finished!')

    def load_pretrained(self, pretrained):
        logging.info('Loading {} pretrained weights from: {}'.format(self.arch, pretrained))
        if self.arch[:3] == 'vgg':
            model_emotion = vgg_utils.VGG_Net(vgg_utils.VGG_Face_torch)
            model_emotion.load_state_dict(torch.load(pretrained))
            model_before_dict = model_emotion.state_dict()
            table_emotion = [0,2,5,7,10,12,14,17,19,21,24,26,28]
            idx = 0
            idxx = 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) and idx < len(table_emotion):
                    weight_name = "pre_model."+str(table_emotion[idx])+".weight"
                    bias_name   = "pre_model."+str(table_emotion[idx])+".bias"
                    assert m.weight.size() == model_before_dict[weight_name].size()
                    assert m.bias.size() == model_before_dict[bias_name].size()
                    m.weight.data = model_before_dict[weight_name]
                    m.bias.data = model_before_dict[bias_name]
                    idx = idx+1
                if isinstance(m, nn.Linear) and idxx < 1:
                    weight_name = "pre_model.32.1.weight"
                    bias_name   = "pre_model.32.1.bias"
                    assert m.weight.size() == model_before_dict[weight_name].size()
                    assert m.bias.size() == model_before_dict[bias_name].size()
                    m.weight.data = model_before_dict[weight_name]
                    m.bias.data = model_before_dict[bias_name]
                    idxx = idxx+1
        elif self.arch[:6] == 'resnet':
            model_emotion = resnet_utils.resnet_face()
            model_emotion.load_state_dict(torch.load(pretrained))
            model_before_dict = model_emotion.state_dict()
            model_dict = self.model.state_dict()

            model_before_dict = {k: v for k, v in model_before_dict.items() if k in model_dict and model_before_dict[k].size() == model_dict[k].size()}
            model_dict.update(model_before_dict)

            self.model.load_state_dict(model_dict)
        elif self.arch == 'c3d':
            corresp_name = {
                            # Conv1
                            "features.0.weight": "conv1.weight",
                            "features.0.bias": "conv1.bias",
                            # Conv2
                            "features.3.weight": "conv2.weight",
                            "features.3.bias": "conv2.bias",
                            # Conv3a
                            "features.6.weight": "conv3a.weight",
                            "features.6.bias": "conv3a.bias",
                            # Conv3b
                            "features.8.weight": "conv3b.weight",
                            "features.8.bias": "conv3b.bias",
                            # Conv4a
                            "features.11.weight": "conv4a.weight",
                            "features.11.bias": "conv4a.bias",
                            # Conv4b
                            "features.13.weight": "conv4b.weight",
                            "features.13.bias": "conv4b.bias",
                            # Conv5a
                            "features.16.weight": "conv5a.weight",
                            "features.16.bias": "conv5a.bias",
                             # Conv5b
                            "features.18.weight": "conv5b.weight",
                            "features.18.bias": "conv5b.bias",
                            # fc6
                            "classifier.0.weight": "fc6.weight",
                            "classifier.0.bias": "fc6.bias",
                            # fc7
                            "classifier.3.weight": "fc7.weight",
                            "classifier.3.bias": "fc7.bias",
                            }

            p_dict = torch.load(pretrained)
            s_dict = self.model.state_dict()
            for name in p_dict:
                if name not in corresp_name:
                    continue
                s_dict[corresp_name[name]] = p_dict[name]
            self.model.load_state_dict(s_dict)
        else:
            logging.info('!!!Do not support {} loading weights'.format(self.arch))
            return
            
        logging.info('Finish Loading!')


def build_model(arch='vgg_lstm', num_classes=8):
    assert arch in ['vgg_gru', 'vgg_lstm', 'vgg_bilstm', 'vgg_gru_nonlocal', 'vgg_bilstm_nonlocal', \
        'resnet_gru', 'resnet_lstm', 'resnet_bilstm', 'resnet_bilstm_nonlocal', 'resnet_gru_nonlocal',\
        'c3d']

    return EmotionNet(arch, num_classes)