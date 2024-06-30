import matplotlib.pyplot as plt
import numpy as np
import torch

def save_comparison_images(content, target, style, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 9))

    pic = content.cpu().squeeze().numpy().transpose((1, 2, 0))
    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
    ax[0].imshow(pic)
    ax[0].set_title('Content picture', fontweight='bold')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    pic = torch.sigmoid(target).cpu().detach().squeeze().numpy().transpose((1, 2, 0))
    ax[1].imshow(pic)
    ax[1].set_title('Target picture', fontweight='bold')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    pic = style.cpu().squeeze().numpy().transpose((1, 2, 0))
    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
    ax[2].imshow(pic, aspect=.6)
    ax[2].set_title('Style picture', fontweight='bold')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    # Save the figure 
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)


def save_image(tensor, path):
    # Convert tensor to numpy array
    img = tensor.cpu().detach().squeeze().numpy().transpose((1, 2, 0))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0, 1]
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    # Save the image
    plt.imsave(path, img)