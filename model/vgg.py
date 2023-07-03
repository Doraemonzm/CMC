# '''VGG for CIFAR10. FC layers are removed.
# (c) YANG, Wei
# '''
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
#
#
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
# }
#
#
# class VGG(nn.Module):
#
#     def __init__(self, cfg, batch_norm=False, num_classes=1000):
#         super(VGG, self).__init__()
#         self.block0 = self._make_layers(cfg[0], batch_norm, 3)
#         self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
#         self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
#         self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
#         self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
#
#         self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
#         # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.classifier = nn.Linear(512, num_classes)
#         self._initialize_weights()
#
#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.block0)
#         feat_m.append(self.pool0)
#         feat_m.append(self.block1)
#         feat_m.append(self.pool1)
#         feat_m.append(self.block2)
#         feat_m.append(self.pool2)
#         feat_m.append(self.block3)
#         feat_m.append(self.pool3)
#         feat_m.append(self.block4)
#         feat_m.append(self.pool4)
#         return feat_m
#
#     def get_bn_before_relu(self):
#         bn1 = self.block1[-1]
#         bn2 = self.block2[-1]
#         bn3 = self.block3[-1]
#         bn4 = self.block4[-1]
#         return [bn1, bn2, bn3, bn4]
#
#     def forward(self, x, is_feat=False, preact=False):
#         h = x.shape[2]
#         x = F.relu(self.block0(x))
#         f0 = x
#         x = self.pool0(x)
#         x = self.block1(x)
#         f1_pre = x
#         x = F.relu(x)
#         f1 = x
#         x = self.pool1(x)
#         x = self.block2(x)
#         f2_pre = x
#         x = F.relu(x)
#         f2 = x
#         x = self.pool2(x)
#         x = self.block3(x)
#         f3_pre = x
#         x = F.relu(x)
#         f3 = x
#         if h == 64:
#             x = self.pool3(x)
#         x = self.block4(x)
#         f4_pre = x
#         x = F.relu(x)
#         f4 = x
#         x = self.pool4(x)
#         x = x.view(x.size(0), -1)
#         f5 = x
#         x = self.classifier(x)
#
#         if is_feat:
#             if preact:
#                 return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
#             else:
#                 return [f0, f1, f2, f3, f4, f5], x
#         else:
#             return x
#
#     @staticmethod
#     def _make_layers(cfg, batch_norm=False, in_channels=3):
#         layers = []
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 if batch_norm:
#                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#                 else:
#                     layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         layers = layers[:-1]
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#
# cfg = {
#     'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
#     'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
#     'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
#     'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
#     'S': [[64], [128], [256], [512], [512]],
# }
#
#
# def vgg8(**kwargs):
#     """VGG 8-layer model (configuration "S")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['S'], **kwargs)
#     return model
#
#
# def vgg8_bn(**kwargs):
#     """VGG 8-layer model (configuration "S")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['S'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg11(**kwargs):
#     """VGG 11-layer model (configuration "A")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['A'], **kwargs)
#     return model
#
#
# def vgg11_bn(**kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     model = VGG(cfg['A'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg13(**kwargs):
#     """VGG 13-layer model (configuration "B")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['B'], **kwargs)
#     return model
#
#
# def vgg13_bn(**kwargs):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     model = VGG(cfg['B'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg16(**kwargs):
#     """VGG 16-layer model (configuration "D")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['D'], **kwargs)
#     return model
#
#
# def vgg16_bn(**kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     model = VGG(cfg['D'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg19(**kwargs):
#     """VGG 19-layer model (configuration "E")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['E'], **kwargs)
#     return model
#
#
# def vgg19_bn(**kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     model = VGG(cfg['E'], batch_norm=True, **kwargs)
#     return model
#
#
# if __name__ == '__main__':
#     import torch
#
#     x = torch.randn(2, 3, 32, 32)
#     net = vgg19_bn(num_classes=100)
#     feats, logit = net(x, is_feat=True, preact=True)
#
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#
#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')



import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feature= torch.flatten(x, 1)
        x = self.classifier(feature)
        return feature, x


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    # if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    if pretrained:
        pretrain_dict = load_state_dict_from_url(model_urls[arch],
                                                 progress=progress)
        model_dict = model.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = vgg19(pretrained=True,num_classes=20)
    feats, logit = net(x)
    print(feats.size())
    print(logit.size())
