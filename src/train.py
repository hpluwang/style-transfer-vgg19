import torch
from .loss import compute_content_loss, compute_style_loss
from .model import get_feature_maps, gram_matrix

def train_style_transfer(vggnet, img4content, img4style, img4target, numepochs, styleScaling, layers4content, layers4style, weights4style, device):
    content_feature_maps, _ = get_feature_maps(img4content, vggnet)
    style_feature_maps, _ = get_feature_maps(img4style, vggnet)

    target = img4target.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.RMSprop([target], lr=0.005)

    for epochi in range(numepochs):
        target_feature_maps, _ = get_feature_maps(target, vggnet)
        content_loss = compute_content_loss(target_feature_maps, content_feature_maps, layers4content)
        style_loss = compute_style_loss(target_feature_maps, style_feature_maps, layers4style, weights4style, gram_matrix)
        combiloss = styleScaling * style_loss + content_loss

        optimizer.zero_grad()
        combiloss.backward()
        optimizer.step()

    return target
