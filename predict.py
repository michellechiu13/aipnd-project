import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import models
from PIL import Image


def load_checkpoint():
    checkpoint = torch.load('checkpoint.pth')

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = checkpoint['epochs']

    return model, optimizer, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pil_image = Image.open(image_path)
    
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256))
    else:
        pil_image.thumbnail((256, 10000))
        
    left_margin = (pil_image.width - 224) / 2
    bottom_margin = (pil_image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    np_image = np.array(pil_image) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        show_image = True
        fig, ax = plt.subplots()
    else:
        show_image = False
    
    image = image.transpose((1, 2, 0))
    
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = image * std + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img = process_image(image_path).unsqueeze(0)
    model.to(device)
    img = img.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    model.train()
    
    indices={val: key for key, val in model.class_to_idx.items()}
    top_labels = [indices[ind] for ind in top_class[0].cpu().detach().numpy()]
    top_p = top_p[0].cpu().detach().numpy()
    top_class = top_labels
    return top_p, top_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="flowers/test/1/image_06764.jpg")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model, optimizer, epochs = load_checkpoint(args.checkpoint)
    top_p, top_class = predict(args.path, model, args.top_k)

    top_flowers=[cat_to_name[key] for key in top_class]

    print(top_flowers)
    print(top_p)