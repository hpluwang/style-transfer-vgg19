# Style Transfer Project

This repository contains code for performing neural style transfer using VGG19 model in PyTorch.

## Overview

Neural style transfer is a technique that allows you to apply the artistic style of one image to the content of another image. In this project, we use the VGG19 convolutional neural network, pretrained on ImageNet, to extract feature maps. These feature maps are used to calculate style and content losses, which are then optimized to generate a new image that combines the content of one image with the style of another.

## Files Structure

The project is organized as follows:

```
style_transfer_project/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── loss.py
│   └── train.py
├── notebooks/
│   ├── style_transfer.ipynb
├── images/
│   ├── content/
│      ├── kanglasha.JPG
│   ├── style/
│      ├── the_scream.jpg
├── output/
│   ├── comparison-image.png
│   ├── output-image.png
├── main.py
└── requirements.txt
```

- **`src/`**: Contains Python modules for dataset handling, model definition (`model.py`), utility functions (`utils.py`), loss functions (`loss.py`), and training script (`train.py`).
- **`notebooks/`**: Contains Jupyter notebook (`StyleTransfer.ipynb`) demonstrating style transfer process.
- **`images/`**: Directory for storing input images (`images/content/kanglasha.JPG` and `images/style/the_scream.jpg`), output images (`output/output-image.png`, `output/comparison-image.png`), and any other image files.
- **`main.py`**: Main script for executing the style transfer process and saving output images.
- **`requirements.txt`**: List of Python packages required to run the project.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hpluwang/style-transfer-vgg19.git
   cd style-transfer-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Ensure you have Python installed, preferably Python 3.x.

2. Navigate to the repository directory:
   ```bash
   cd style-transfer-vgg19
   ```

3. Run the style transfer process:
   ```bash
   python main.py
   ```

4. The output image (`output-image.png`) and comparison image (`comparison-image.png`) will be saved in the `output/` directory.

## Customization

- **Adjusting Parameters**: You can modify parameters like `numepochs`, `styleScaling`, `layers4content`, `layers4style`, and `weights4style` in `main.py` to experiment with different style transfer results.
- **Adding New Images**: Place new content and style images in the `images/` directory and update paths in `main.py` accordingly.

## License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## Acknowledgments

- This project uses the VGG19 model pretrained on ImageNet from PyTorch torchvision.
- Inspired by neural style transfer techniques and tutorials.
