# ArtGAN: Artistic Style Transfer

ArtGAN is a neural style transfer application that transforms content images by applying the style of a different image. The application is implemented using PyTorch and Streamlit for an easy-to-use web interface.

## Features
- Upload content and style images to create art using neural style transfer
- Choose from various style layers and weights for custom results
- Real-time, GPU-accelerated transformations

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ArtGAN-Style-Transfer.git
    cd ArtGAN-Style-Transfer
    ```
2. Install dependencies:
    ```bash
    pip install -r src/requirements.txt
    ```

## Usage
Run the Streamlit application with:
```bash
streamlit run src/main.py
