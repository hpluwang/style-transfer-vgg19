import torchvision.transforms as T
import imageio.v2 as imageio
import torch

def load_and_transform_image(image_path, device, size=256):
    image = imageio.imread(image_path)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image
