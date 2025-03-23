# This Artwork Doesn’t Exist

This repository contains the implementation and deployment of a Deep Convolutional Generative Adversarial Network (DCGAN) that generates novel artworks using a curated subset of the WikiArt dataset. Developed as part of Assignment 2 for the Generative AI & Applications course at SEECS, NUST by Basharat Ali, this project explores the intersection of technology and creativity.

## Project Overview

- **Objective:**  
  Train a DCGAN to learn artistic features from a diverse collection of artworks and generate original images.

- **Dataset:**  
  A curated subset of 61,000 images from the WikiArt dataset was used. The images were preprocessed by:
  - Resizing to 64×64 pixels
  - Normalizing pixel values to a range of [-1, 1]

- **Model Training:**  
  The DCGAN was trained using the following parameters:
  - **Latent Dimension:** 100  
  - **Batch Size:** 1024  
  - **Number of Epochs:** 150  
  - **Image Size:** 64×64  
  - **Learning Rate:** 0.0002  
  Training was performed on Kaggle’s T4x2 GPUs.

## Deployment

The trained model is deployed on [Huggingface Spaces](https://huggingface.co/spaces/basharatwali/Wiki_ArtGAN), where an interactive Gradio interface allows users to generate and explore artwork on the fly. The deployment code is available in the `HF_App` folder.

## Repository Structure

- **HF_App:** Contains the code for model inference and the Gradio interface used for deploying the application on Huggingface Spaces.
- **Training:** Training and Inference code is in Training folder.

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BasharatWali/Wiki_ArtGAN.git
