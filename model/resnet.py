# #
# # """
# # Code
# # source: https: // github.com / pytorch / vision
# # """
# from __future__ import division, absolute_import
# import torch.utils.model_zoo as model_zoo
# from torch import nn
#
# __all__ = [
#     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#     'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512'
# ]
#
# model_urls = {
#     'resnet18':
#     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34':
#     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50':
#     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101':
#     'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152':
#     'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d':
#     'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d':
#     'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
# }
#
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """
# 3x3
# convolution
# with padding
# """
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation
#     )
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=1, stride=stride, bias=False
#     )
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None,
#         is_last=False
#     ):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64'
#             )
#         if dilation > 1:
#             raise NotImplementedError(
#                 "Dilation > 1 not supported in BasicBlock"
#             )
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.is_last = is_last
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         preact = out
#         out = self.relu(out)
#
#         if self.is_last:
#             return out, preact
#         else:
#             return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None,
#         is_last=False
#     ):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width/64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.is_last = is_last
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         preact = out
#         out = self.relu(out)
#         if self.is_last:
#             return out, preact
#         else:
#             return out
#
#
#
# class ResNet(nn.Module):
#     """Residual network.
# Reference:
# - He et al.Deep Residual Learning for Image Recognition.CVPR 2016.
# - Xie et al.Aggregated Residual Transformations for Deep Neural Networks.CVPR 2017.
# Public keys:
# - ``resnet18``: ResNet18.
# - ``resnet34``: ResNet34.
# - ``resnet50``: ResNet50.
# - ``resnet101``: ResNet101.
# - ``resnet152``: ResNet152.
# - ``resnext50_32x4d``: ResNeXt50.
# - ``resnext101_32x8d``: ResNeXt101.
# - ``resnet50_fc512``: ResNet50 + FC.
# """
#
#     def __init__(
#         self,
#         num_classes,
#         block,
#         layers,
#         zero_init_residual=False,
#         groups=1,
#         width_per_group=64,
#         replace_stride_with_dilation=None,
#         norm_layer=None,
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     ):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.feature_dim = 512 * block.expansion
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 "or a 3-element tuple, got {}".
#                 format(replace_stride_with_dilation)
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(
#             3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
#         )
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(
#             block,
#             128,
#             layers[1],
#             stride=2,
#             dilate=replace_stride_with_dilation[0]
#         )
#         self.layer3 = self._make_layer(
#             block,
#             256,
#             layers[2],
#             stride=2,
#             dilate=replace_stride_with_dilation[1]
#         )
#         self.layer4 = self._make_layer(
#             block,
#             512,
#             layers[3],
#             stride=last_stride,
#             dilate=replace_stride_with_dilation[2]
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = self._construct_fc_layer(
#             fc_dims, 512 * block.expansion, dropout_p
#         )
#         self.fc=nn.Linear(512 * block.expansion, num_classes)
#         self.classifier = nn.Linear(self.feature_dim, num_classes)
#         #
#         self._init_params()
#
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups,
#                 self.base_width, previous_dilation, norm_layer, is_last=(blocks == 1)
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                     is_last=(i == blocks - 1)
#                 )
#             )
#
#         return nn.Sequential(*layers)
#
#     def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
#         """Constructs fully connected layer
#         Args:
#             fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
#             input_dim (int): input dimension
#             dropout_p (float): dropout probability, if None, dropout is unused
#         """
#         if fc_dims is None:
#             self.feature_dim = input_dim
#             return None
#
#         assert isinstance(
#             fc_dims, (list, tuple)
#         ), 'fc_dims must be either list or tuple, but got {}'.format(
#             type(fc_dims)
#         )
#
#         layers = []
#         for dim in fc_dims:
#             layers.append(nn.Linear(input_dim, dim))
#             layers.append(nn.BatchNorm1d(dim))
#             layers.append(nn.ReLU(inplace=True))
#             if dropout_p is not None:
#                 layers.append(nn.Dropout(p=dropout_p))
#             input_dim = dim
#
#         self.feature_dim = fc_dims[-1]
#
#         return nn.Sequential(*layers)
#
#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu'
#                 )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.conv1)
#         feat_m.append(self.bn1)
#         feat_m.append(self.relu)
#         # feat_m.append(self.maxpool)
#         feat_m.append(self.layer1)
#         feat_m.append(self.layer2)
#         feat_m.append(self.layer3)
#         feat_m.append(self.layer4)
#         return feat_m
#     # def featuremaps(self, x):
#     #     x = self.conv1(x)
#     #     x = self.bn1(x)
#     #     x = self.relu(x)
#     #     x = self.maxpool(x)
#     #     x = self.layer1(x)
#     #     x = self.layer2(x)
#     #     x = self.layer3(x)
#     #     x = self.layer4(x)
#     #     return x
#
#     def forward(self, x,is_feat=False):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.maxpool(x)
#         f0=x
#
#         x,f1_pre = self.layer1(x)
#         f1=x
#         x,f2_pre = self.layer2(x)
#         f2=x
#         x,f3_pre = self.layer3(x)
#         f3=x
#         x,f4_pre = self.layer4(x)
#         f4=x
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         f5=x
#         x = self.fc(x)
#         if is_feat:
#             return f5, x
#         else:
#             return x
#
# def init_pretrained_weights(model, model_url):
# # Initializes model with pretrained weights.
# # Layers that don't match with pretrained layers in name or size are kept unchanged.
#
#     pretrain_dict = model_zoo.load_url(model_url)
#     model_dict = model.state_dict()
#     pretrain_dict = {
#         k: v
#         for k, v in pretrain_dict.items()
#         if k in model_dict and model_dict[k].size() == v.size()
#     }
#     model_dict.update(pretrain_dict)
#     model.load_state_dict(model_dict)
#
#
# """
# ResNet
# """
#
#
# def resnet18(num_classes,pretrained=False, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=BasicBlock,
#         layers=[2, 2, 2, 2],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet18'])
#     return model
#
#
# def resnet34(num_classes,pretrained=False, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=BasicBlock,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet34'])
#     return model
#
#
# def resnet50(num_classes, pretrained=True,**kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet50'])
#     return model
#
#
# def resnet101(num_classes, pretrained=False,**kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 4, 23, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet101'])
#     return model
#
#
# def resnet152(num_classes, pretrained=False,**kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 8, 36, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet152'])
#     return model
#
#
# """
# ResNeXt
# """
#
#
# def resnext50_32x4d(num_classes, pretrained=False, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         groups=32,
#         width_per_group=4,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnext50_32x4d'])
#     return model
#
#
# def resnext101_32x8d(num_classes, pretrained=False, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 4, 23, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         groups=32,
#         width_per_group=8,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnext101_32x8d'])
#
#     return model
#
#
# """
# ResNet + FC
# """
#
#
# def resnet50_fc512(num_classes, pretrained=False,**kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=1,
#         fc_dims=[512],
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet50'])
#     return model
#
# if __name__ == '__main__':
#     import torch
#
#     x = torch.randn(2, 3, 32, 32)
#     net1 = resnet18(num_classes=20)
#     net2 = resnet50(num_classes=20)
#     feats1, logit1 = net1(x, is_feat=True)
#     feats2, logit2 = net2(x, is_feat=True)
#     print("resnet18:------------")
#     for f in feats1:
#         print(f.shape, f.min().item())
#     print(logit1.shape)
#     print("resnet50:------------")
#     for f in feats2:
#         print(f.shape, f.min().item())
#     print(logit2.shape)
# #
# #
# #
# #
# # # """
# # # Code
# # # source: https: // github.com / pytorch / vision
# # # """
# # # from __future__ import division, absolute_import
# # # import torch.utils.model_zoo as model_zoo
# # # from torch import nn
# #
# # # __all__ = [
# # #     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# # #     'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512'
# # # ]
# #
# # # model_urls = {
# # #     'resnet18':
# # #     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
# # #     'resnet34':
# # #     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
# # #     'resnet50':
# # #     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
# # #     'resnet101':
# # #     'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
# # #     'resnet152':
# # #     'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# # #     'resnext50_32x4d':
# # #     'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
# # #     'resnext101_32x8d':
# # #     'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
# # # }
# #
# #
# # # class Normalize(nn.Module):
# #
# # #     def __init__(self, power=2):
# # #         super(Normalize, self).__init__()
# # #         self.power = power
# #
# # #     def forward(self, x):
# # #         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
# # #         out = x.div(norm)
# # #         return out
# #
# #
# #
# # # def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
# # #     """
# # # 3x3
# # # convolution
# # # with padding
# # # """
# # #     return nn.Conv2d(
# # #         in_planes,
# # #         out_planes,
# # #         kernel_size=3,
# # #         stride=stride,
# # #         padding=dilation,
# # #         groups=groups,
# # #         bias=False,
# # #         dilation=dilation
# # #     )
# #
# #
# # # def conv1x1(in_planes, out_planes, stride=1):
# # #     """1x1 convolution"""
# # #     return nn.Conv2d(
# # #         in_planes, out_planes, kernel_size=1, stride=stride, bias=False
# # #     )
# #
# #
# # # class BasicBlock(nn.Module):
# # #     expansion = 1
# #
# # #     def __init__(
# # #         self,
# # #         inplanes,
# # #         planes,
# # #         stride=1,
# # #         downsample=None,
# # #         groups=1,
# # #         base_width=64,
# # #         dilation=1,
# # #         norm_layer=None,
# # #         is_last=False
# # #     ):
# # #         super(BasicBlock, self).__init__()
# # #         if norm_layer is None:
# # #             norm_layer = nn.BatchNorm2d
# # #         if groups != 1 or base_width != 64:
# # #             raise ValueError(
# # #                 'BasicBlock only supports groups=1 and base_width=64'
# # #             )
# # #         if dilation > 1:
# # #             raise NotImplementedError(
# # #                 "Dilation > 1 not supported in BasicBlock"
# # #             )
# # #         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
# # #         self.conv1 = conv3x3(inplanes, planes, stride)
# # #         self.bn1 = norm_layer(planes)
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.conv2 = conv3x3(planes, planes)
# # #         self.bn2 = norm_layer(planes)
# # #         self.downsample = downsample
# # #         self.stride = stride
# # #         self.is_last = is_last
# #
# # #     def forward(self, x):
# # #         identity = x
# #
# # #         out = self.conv1(x)
# # #         out = self.bn1(out)
# # #         out = self.relu(out)
# #
# # #         out = self.conv2(out)
# # #         out = self.bn2(out)
# #
# # #         if self.downsample is not None:
# # #             identity = self.downsample(x)
# #
# # #         out += identity
# # #         preact = out
# # #         out = self.relu(out)
# #
# # #         if self.is_last:
# # #             return out, preact
# # #         else:
# # #             return out
# #
# #
# # # class Bottleneck(nn.Module):
# # #     expansion = 4
# #
# # #     def __init__(
# # #         self,
# # #         inplanes,
# # #         planes,
# # #         stride=1,
# # #         downsample=None,
# # #         groups=1,
# # #         base_width=64,
# # #         dilation=1,
# # #         norm_layer=None,
# # #         is_last=False
# # #     ):
# # #         super(Bottleneck, self).__init__()
# # #         if norm_layer is None:
# # #             norm_layer = nn.BatchNorm2d
# # #         width = int(planes * (base_width/64.)) * groups
# # #         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
# # #         self.conv1 = conv1x1(inplanes, width)
# # #         self.bn1 = norm_layer(width)
# # #         self.conv2 = conv3x3(width, width, stride, groups, dilation)
# # #         self.bn2 = norm_layer(width)
# # #         self.conv3 = conv1x1(width, planes * self.expansion)
# # #         self.bn3 = norm_layer(planes * self.expansion)
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.downsample = downsample
# # #         self.stride = stride
# # #         self.is_last = is_last
# #
# # #     def forward(self, x):
# # #         identity = x
# #
# # #         out = self.conv1(x)
# # #         out = self.bn1(out)
# # #         out = self.relu(out)
# #
# # #         out = self.conv2(out)
# # #         out = self.bn2(out)
# # #         out = self.relu(out)
# #
# # #         out = self.conv3(out)
# # #         out = self.bn3(out)
# #
# # #         if self.downsample is not None:
# # #             identity = self.downsample(x)
# #
# # #         out += identity
# # #         preact = out
# # #         out = self.relu(out)
# # #         if self.is_last:
# # #             return out, preact
# # #         else:
# # #             return out
# #
# #
# #
# # # class ResNet(nn.Module):
# # #     """Residual network.
# # # Reference:
# # # - He et al.Deep Residual Learning for Image Recognition.CVPR 2016.
# # # - Xie et al.Aggregated Residual Transformations for Deep Neural Networks.CVPR 2017.
# # # Public keys:
# # # - ``resnet18``: ResNet18.
# # # - ``resnet34``: ResNet34.
# # # - ``resnet50``: ResNet50.
# # # - ``resnet101``: ResNet101.
# # # - ``resnet152``: ResNet152.
# # # - ``resnext50_32x4d``: ResNeXt50.
# # # - ``resnext101_32x8d``: ResNeXt101.
# # # - ``resnet50_fc512``: ResNet50 + FC.
# # # """
# #
# # #     def __init__(
# # #         self,
# # #         num_classes,
# # #         block,
# # #         layers,
# # #         zero_init_residual=False,
# # #         groups=1,
# # #         width_per_group=64,
# # #         replace_stride_with_dilation=None,
# # #         norm_layer=None,
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     ):
# # #         super(ResNet, self).__init__()
# # #         if norm_layer is None:
# # #             norm_layer = nn.BatchNorm2d
# # #         self._norm_layer = norm_layer
# # #         self.feature_dim = 512 * block.expansion
# # #         self.inplanes = 64
# # #         self.dilation = 1
# # #         if replace_stride_with_dilation is None:
# # #             # each element in the tuple indicates if we should replace
# # #             # the 2x2 stride with a dilated convolution instead
# # #             replace_stride_with_dilation = [False, False, False]
# # #         if len(replace_stride_with_dilation) != 3:
# # #             raise ValueError(
# # #                 "replace_stride_with_dilation should be None "
# # #                 "or a 3-element tuple, got {}".
# # #                 format(replace_stride_with_dilation)
# # #             )
# # #         self.groups = groups
# # #         self.base_width = width_per_group
# # #         self.conv1 = nn.Conv2d(
# # #             3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
# # #         )
# # #         self.bn1 = norm_layer(self.inplanes)
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# # #         self.layer1 = self._make_layer(block, 64, layers[0])
# # #         self.layer2 = self._make_layer(
# # #             block,
# # #             128,
# # #             layers[1],
# # #             stride=2,
# # #             dilate=replace_stride_with_dilation[0]
# # #         )
# # #         self.layer3 = self._make_layer(
# # #             block,
# # #             256,
# # #             layers[2],
# # #             stride=2,
# # #             dilate=replace_stride_with_dilation[1]
# # #         )
# # #         self.layer4 = self._make_layer(
# # #             block,
# # #             512,
# # #             layers[3],
# # #             stride=last_stride,
# # #             dilate=replace_stride_with_dilation[2]
# # #         )
# # #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# # #         # self.avgpool = nn.AvgPool2d(7, stride=1)
# # #         self.fc = self._construct_fc_layer(
# # #             fc_dims, 512 * block.expansion, dropout_p
# # #         )
# # #         self.fc=nn.Linear(512 * block.expansion, 128)
# # #         self.l2norm = Normalize(2)
# # #         self.classifier = nn.Linear(self.feature_dim, num_classes)
# #
# # #         self._init_params()
# #
# # #         if zero_init_residual:
# # #             for m in self.modules():
# # #                 if isinstance(m, Bottleneck):
# # #                     nn.init.constant_(m.bn3.weight, 0)
# # #                 elif isinstance(m, BasicBlock):
# # #                     nn.init.constant_(m.bn2.weight, 0)
# #
# # #         # Zero-initialize the last BN in each residual branch,
# # #         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
# # #         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
# # #         for m in self.modules():
# # #             if isinstance(m, nn.Conv2d):
# # #                 nn.init.kaiming_normal_(
# # #                     m.weight, mode='fan_out', nonlinearity='relu')
# # #             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
# # #                 nn.init.constant_(m.weight, 1)
# # #                 nn.init.constant_(m.bias, 0)
# #
# #
# # #     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
# # #         norm_layer = self._norm_layer
# # #         downsample = None
# # #         previous_dilation = self.dilation
# # #         if dilate:
# # #             self.dilation *= stride
# # #             stride = 1
# # #         if stride != 1 or self.inplanes != planes * block.expansion:
# # #             downsample = nn.Sequential(
# # #                 conv1x1(self.inplanes, planes * block.expansion, stride),
# # #                 norm_layer(planes * block.expansion),
# # #             )
# #
# # #         layers = []
# # #         layers.append(
# # #             block(
# # #                 self.inplanes, planes, stride, downsample, self.groups,
# # #                 self.base_width, previous_dilation, norm_layer, is_last=(blocks == 1)
# # #             )
# # #         )
# # #         self.inplanes = planes * block.expansion
# # #         for i in range(1, blocks):
# # #             layers.append(
# # #                 block(
# # #                     self.inplanes,
# # #                     planes,
# # #                     groups=self.groups,
# # #                     base_width=self.base_width,
# # #                     dilation=self.dilation,
# # #                     norm_layer=norm_layer,
# # #                     is_last=(i == blocks - 1)
# # #                 )
# # #             )
# #
# # #         return nn.Sequential(*layers)
# #
# # #     def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
# # #         """Constructs fully connected layer
# # #         Args:
# # #             fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
# # #             input_dim (int): input dimension
# # #             dropout_p (float): dropout probability, if None, dropout is unused
# # #         """
# # #         if fc_dims is None:
# # #             self.feature_dim = input_dim
# # #             return None
# #
# # #         assert isinstance(
# # #             fc_dims, (list, tuple)
# # #         ), 'fc_dims must be either list or tuple, but got {}'.format(
# # #             type(fc_dims)
# # #         )
# #
# # #         layers = []
# # #         for dim in fc_dims:
# # #             layers.append(nn.Linear(input_dim, dim))
# # #             layers.append(nn.BatchNorm1d(dim))
# # #             layers.append(nn.ReLU(inplace=True))
# # #             if dropout_p is not None:
# # #                 layers.append(nn.Dropout(p=dropout_p))
# # #             input_dim = dim
# #
# # #         self.feature_dim = fc_dims[-1]
# #
# # #         return nn.Sequential(*layers)
# #
# # #     def _init_params(self):
# # #         for m in self.modules():
# # #             if isinstance(m, nn.Conv2d):
# # #                 nn.init.kaiming_normal_(
# # #                     m.weight, mode='fan_out', nonlinearity='relu'
# # #                 )
# # #                 if m.bias is not None:
# # #                     nn.init.constant_(m.bias, 0)
# # #             elif isinstance(m, nn.BatchNorm2d):
# # #                 nn.init.constant_(m.weight, 1)
# # #                 nn.init.constant_(m.bias, 0)
# # #             elif isinstance(m, nn.BatchNorm1d):
# # #                 nn.init.constant_(m.weight, 1)
# # #                 nn.init.constant_(m.bias, 0)
# # #             elif isinstance(m, nn.Linear):
# # #                 nn.init.normal_(m.weight, 0, 0.01)
# # #                 if m.bias is not None:
# # #                     nn.init.constant_(m.bias, 0)
# #
# # #     def get_feat_modules(self):
# # #         feat_m = nn.ModuleList([])
# # #         feat_m.append(self.conv1)
# # #         feat_m.append(self.bn1)
# # #         feat_m.append(self.relu)
# # #         # feat_m.append(self.maxpool)
# # #         feat_m.append(self.layer1)
# # #         feat_m.append(self.layer2)
# # #         feat_m.append(self.layer3)
# # #         feat_m.append(self.layer4)
# # #         return feat_m
# # #     # def featuremaps(self, x):
# # #     #     x = self.conv1(x)
# # #     #     x = self.bn1(x)
# # #     #     x = self.relu(x)
# # #     #     x = self.maxpool(x)
# # #     #     x = self.layer1(x)
# # #     #     x = self.layer2(x)
# # #     #     x = self.layer3(x)
# # #     #     x = self.layer4(x)
# # #     #     return x
# #
# # #     def forward(self, x):
# # #         x = self.conv1(x)
# # #         x = self.bn1(x)
# # #         x = self.relu(x)
# # #         x = self.maxpool(x)
# # #         f0=x
# #
# # #         x,f1_pre = self.layer1(x)
# # #         f1=x
# # #         x,f2_pre = self.layer2(x)
# # #         f2=x
# # #         x,f3_pre = self.layer3(x)
# # #         f3=x
# # #         x,f4_pre = self.layer4(x)
# # #         f4=x
# # #         x = self.avgpool(x)
# # #         x = x.view(x.size(0), -1)
# # #         f5=x
# # #         final = self.fc(f5)
# # #         feature_128=self.l2norm(final)
# # #         logit=self.classifier(f5)
# # #         return feature_128,logit
# # #         # if is_feat:
# # #         #     if preact:
# # #         #         return [f0, f1_pre, f2_pre, f3_pre, f4_pre,f5], x
# # #         #     else:
# # #         #         return [f0, f1, f2, f3, f4,f5], x
# # #         # else:
# # #         #     return x
# #
# # # def init_pretrained_weights(model, model_url):
# # # # Initializes model with pretrained weights.
# # # # Layers that don't match with pretrained layers in name or size are kept unchanged.
# #
# # #     pretrain_dict = model_zoo.load_url(model_url)
# # #     model_dict = model.state_dict()
# # #     pretrain_dict = {
# # #         k: v
# # #         for k, v in pretrain_dict.items()
# # #         if k in model_dict and model_dict[k].size() == v.size()
# # #     }
# # #     model_dict.update(pretrain_dict)
# # #     model.load_state_dict(model_dict)
# #
# #
# # # """
# # # ResNet
# # # """
# #
# #
# # # def resnet18(num_classes,pretrained=False, **kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=BasicBlock,
# # #         layers=[2, 2, 2, 2],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet18'])
# # #     return model
# #
# #
# # # def resnet34(num_classes,pretrained=False, **kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=BasicBlock,
# # #         layers=[3, 4, 6, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet34'])
# # #     return model
# #
# #
# # # def resnet50(num_classes, pretrained=True,**kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 4, 6, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet50'])
# # #     return model
# #
# #
# # # def resnet101(num_classes, pretrained=False,**kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 4, 23, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet101'])
# # #     return model
# #
# #
# # # def resnet152(num_classes, pretrained=False,**kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 8, 36, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet152'])
# # #     return model
# #
# #
# # # """
# # # ResNeXt
# # # """
# #
# #
# # # def resnext50_32x4d(num_classes, pretrained=False, **kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 4, 6, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         groups=32,
# # #         width_per_group=4,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnext50_32x4d'])
# # #     return model
# #
# #
# # # def resnext101_32x8d(num_classes, pretrained=False, **kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 4, 23, 3],
# # #         last_stride=2,
# # #         fc_dims=None,
# # #         dropout_p=None,
# # #         groups=32,
# # #         width_per_group=8,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnext101_32x8d'])
# #
# # #     return model
# #
# #
# # # """
# # # ResNet + FC
# # # """
# #
# #
# # # def resnet50_fc512(num_classes, pretrained=False,**kwargs):
# # #     model = ResNet(
# # #         num_classes=num_classes,
# # #         block=Bottleneck,
# # #         layers=[3, 4, 6, 3],
# # #         last_stride=1,
# # #         fc_dims=[512],
# # #         dropout_p=None,
# # #         **kwargs
# # #     )
# # #     if pretrained:
# # #         init_pretrained_weights(model, model_urls['resnet50'])
# # #     return model
# #
# # # if __name__ == '__main__':
# # #     import torch
# #
# # #     x = torch.randn(2, 3, 32, 32)
# # #     net1 = resnet18(num_classes=20)
# # #     net2 = resnet50(num_classes=20)
# # #     feats1, logit1 = net1(x)
# # #     feats2, logit2 = net2(x)
# # #     print("resnet18:------------")
# # #     for f in feats1:
# # #         print(f.shape, f.min().item())
# # #     print(logit1.shape)
# # #     print("resnet50:------------")
# # #     for f in feats2:
# # #         print(f.shape, f.min().item())
# # #     print(logit2.shape)
# #
# #
# #
# #
# #
# #
#
#
#
import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)

        return feat, x




def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
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
    #
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnet50(pretrained=True,num_classes=20)
    feats, logit = net(x)
    print(feats.size())
    print(logit.size())

