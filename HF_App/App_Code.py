from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import torch
import torch.nn as nn

latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_map_size=32):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def generate_artwork(generator, latent_dim=latent_dim, device=device, num_images=1):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_images = fake_images * 0.5 + 0.5
    return fake_images.detach().cpu()

def inference_interface(latent_dim=latent_dim, device=device):
    # Create model and load weights
    generator = Generator(latent_dim=latent_dim)
    generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
    
    if isinstance(generator, nn.DataParallel):
        generator = generator.module
    generator.to(device)

    def generate(num_images):
        fake_images = generate_artwork(generator, latent_dim=latent_dim, device=device, num_images=num_images)
        images = [transforms.ToPILImage()(img) for img in fake_images]
        upscaled_images = [img.resize((256, 256), resample=Image.LANCZOS) for img in images]
        return upscaled_images

    demo = gr.Interface(fn=generate,
      inputs=gr.Slider(minimum=1, maximum=9, step=1, value=1, label="Number of Images"),
      outputs=gr.Gallery(label="Generated Artwork", columns=3, height="auto"),
      title="This Artwork Doesn’t Exist",
      description="Generate artwork using our Wiki_ArtGAN."
      )

    return demo

# The key part: launch the Gradio interface when app.py is run
if __name__ == "__main__":
    demo = inference_interface()
    demo.launch()
