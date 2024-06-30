import torch
import torch.nn as nn
import torchvision.models as models

def get_vgg19_model(device):
    vggnet = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    for p in vggnet.parameters():
        p.requires_grad = False
    vggnet.to(device)
    vggnet.eval()
    return vggnet

def get_feature_maps(img, net):
    featuremaps = []
    featurenames = []
    convLayerIdx = 0
    for layernum in range(len(net.features)):
        img = net.features[layernum](img)
        if 'Conv2d' in str(net.features[layernum]):
            featuremaps.append(img)
            featurenames.append('ConvLayer_' + str(convLayerIdx))
            convLayerIdx += 1
    return featuremaps, featurenames

def gram_matrix(M):
    _, chans, height, width = M.shape
    M = M.reshape(chans, height * width)
    gram = torch.mm(M, M.t()) / (chans * height * width)
    return gram
