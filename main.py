import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the images
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    # Scale image
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Discard the transparent, alpha channel (if it exists)
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)


# Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))


# Define content and style layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                  '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


# Define content and style loss
def calculate_content_loss(target_features, content_features):
    return torch.mean((target_features - content_features) ** 2)


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def calculate_style_loss(target_features, style_grams, style_weights):
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2])
    return style_loss


# Neural style transfer function
def neural_style_transfer(content_img, style_img, steps=2000, content_weight=1e4, style_weight=1e2):
    # Load content and style images
    content = load_image(content_img).to(device)
    style = load_image(style_img, shape=content.shape[-2:]).to(device)

    # Load pre-trained VGG19
    vgg = models.vgg19(pretrained=True).features

    # Freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)

    vgg.to(device)

    # Extract content and style features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Create the target image to optimize
    target = content.clone().requires_grad_(True).to(device)

    # Define weights for each style layer
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

    # Setup optimizer
    optimizer = optim.Adam([target], lr=0.003)

    # Train the network
    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = calculate_content_loss(target_features['conv4_2'], content_features['conv4_2'])
        style_loss = calculate_style_loss(target_features, style_grams, style_weights)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % 400 == 0:
            print(f"Step {ii}/{steps}, Total loss: {total_loss.item()}")

    return im_convert(target)


# Streamlit Web App
st.title("ArtGAN: Artistic Style Transfer")

# Upload content and style images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

if content_file is not None and style_file is not None:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    st.image([content_image, style_image], caption=["Content Image", "Style Image"], width=300)

    if st.button("Transform"):
        with st.spinner('Transforming... Please wait...'):
            styled_image = neural_style_transfer(content_file, style_file)
            st.image(styled_image, caption="Styled Image", use_column_width=True)
