import torch

def compute_content_loss(target_feature_maps, content_feature_maps, layers4content):
    content_loss = 0
    for layer_name, target_feature, content_feature in zip(layers4content, target_feature_maps, content_feature_maps):
        content_loss += torch.mean((target_feature - content_feature) ** 2)
    return content_loss

def compute_style_loss(target_feature_maps, style_feature_maps, layers4style, weights4style, gram_matrix):
    style_loss = 0
    for layer_name, target_feature, style_feature, weight in zip(layers4style, target_feature_maps, style_feature_maps, weights4style):
        Gtarget = gram_matrix(target_feature)
        Gstyle = gram_matrix(style_feature)
        style_loss += torch.mean((Gtarget - Gstyle) ** 2) * weight
    return style_loss
