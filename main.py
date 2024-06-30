import torch
from src.dataset import load_and_transform_image
from src.model import get_vgg19_model
from src.utils import save_comparison_images, save_image
from src.train import train_style_transfer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

img4content = load_and_transform_image('images/content/kanglasha.JPG', device)
img4style = load_and_transform_image('images/style/the_scream.jpg', device)
img4target = torch.rand_like(img4content).to(device)

vggnet = get_vgg19_model(device)

layers4content = ['ConvLayer_1', 'ConvLayer_4']
layers4style = ['ConvLayer_1', 'ConvLayer_2', 'ConvLayer_3', 'ConvLayer_4', 'ConvLayer_5']
weights4style = [3, 1, 0.5, 0.2, 0.1] # Adjust style strength (higher -> more style)

numepochs = 1500
styleScaling = 1e6

target = train_style_transfer(vggnet, img4content, img4style, img4target, numepochs, styleScaling, layers4content, layers4style, weights4style, device)

# Save the comparison image
comparison_save_path = 'output/comparison-image.png'
save_comparison_images(img4content, target, img4style, save_path=comparison_save_path)

# Save the output image
save_image(torch.sigmoid(target), 'output/output-image.png')