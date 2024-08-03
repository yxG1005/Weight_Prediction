import os
import torch
import numpy as np
import random


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        
        if self.args.is_training == False and self.args.fusion == "late":
            self.image_model, self.text_model = self._build_model()
            self.image_model = self.image_model.to(self.device)
            self.text_model = self.text_model.to(self.device)
 
        else:
            self.model = self._build_model()
            self.model = self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
