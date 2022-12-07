# %% 

import torch as t
from torch import nn, optim
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Union, Callable, Tuple
import plotly.express as px
import torchinfo
import time
from dataclasses import dataclass
import wandb
from PIL import Image
import pandas as pd

MAIN = __name__ == "__main__"

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# assert str(device) == "cuda:0"

# %%
# ============================================ Data ============================================

if MAIN:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root="./data/mnist/", train=True, transform=transform, download=True)
    testset = datasets.MNIST(root="./data/mnist/", train=False, transform=transform, download=True)

    data_to_plot = dict()
    for data, target in DataLoader(testset, batch_size=1):
        if target.item() not in data_to_plot:
            data_to_plot[target.item()] = data.squeeze()
            if len(data_to_plot) == 10:
                break
    data_to_plot = t.stack([data_to_plot[i] for i in range(10)]).to(t.float).unsqueeze(1)

    loss_fn = nn.MSELoss()

    # print_output_interval = 10

    # epochs = 10

# %%
# ============================================ Autoencoders ============================================


class Autoencoder(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()

        in_features_list = (28*28, 100)
        out_features_list = (100, latent_dim_size)

        encoder = [("rearrange", Rearrange("batch 1 height width -> batch (height width)"))]
        for i, (ic, oc) in enumerate(zip(in_features_list, out_features_list), 1):
            encoder.append((f"fc{i}", nn.Linear(ic, oc)))
            if i < len(in_features_list):
                encoder.append((f"relu{i}", nn.ReLU()))
        self.encoder = nn.Sequential(OrderedDict(encoder))

        decoder = []
        for i, (ic, oc) in enumerate(zip(out_features_list[::-1], in_features_list[::-1]), 1):
            decoder.append((f"fc{i}", nn.Linear(ic, oc)))
            if i < len(in_features_list):
                decoder.append((f"relu{i}", nn.ReLU()))
        decoder.append(("rearrange", Rearrange("batch (height width) -> batch 1 height width", height=28)))
        self.decoder = nn.Sequential(OrderedDict(decoder))

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if MAIN:
    model = Autoencoder(latent_dim_size=5).to(device)
    optimizer = optim.Adam(model.parameters())
    print(torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0)))

# %%

class AutoencoderLarge(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if MAIN:
    model = AutoencoderLarge(latent_dim_size=5)
    optimizer = optim.Adam(model.parameters())
    print(torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0)))


# %%

def show_images(model, data_to_plot, return_arr=False):

    device = next(model.parameters()).device
    data_to_plot = data_to_plot.to(device)
    output = model(data_to_plot)
    if isinstance(output, tuple):
        output = output[0]

    both = t.concat((data_to_plot.squeeze(), output.squeeze()), dim=0).cpu().detach().numpy()
    both = np.clip((both * 0.3081) + 0.1307, 0, 1)

    if return_arr:
        arr = rearrange(both, "(b1 b2) h w -> (b1 h) (b2 w) 1", b1=2)
        return arr

    fig = px.imshow(both, facet_col=0, facet_col_wrap=10, color_continuous_scale="greys_r")
    fig.update_layout(coloraxis_showscale=False).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    for i in range(10):
        fig.layout.annotations[i]["text"] = ""
        fig.layout.annotations[i+10]["text"] = str(i)
    fig.show()


# %%

@dataclass
class AutoencoderArgs():
    latent_dim_size: int = 5
    use_large: bool = True
    epochs: int = 3
    loss_fn: Callable = nn.MSELoss()
    img_size: int = 28
    img_channels: int = 1
    batch_size: int = 512
    track: bool = True
    cuda: bool = True
    seconds_between_image_logs: int = 5
    trainset = trainset
    testset = testset
    data_to_plot: t.Tensor = data_to_plot # Set of images to plot or log


# def train_autoencoder(model, optimizer, loss_fn, trainset, data_to_plot, epochs, batch_size, print_output_interval=15, use_wandb=True):
def train_autoencoder(args: AutoencoderArgs):

    model = AutoencoderLarge(args.latent_dim_size) if args.use_large else Autoencoder(args.latent_dim_size)
    model.to(device).train()
    optimizer = t.optim.Adam(model.parameters())

    t_last = time.time()
    examples_seen = 0

    data_to_plot = args.data_to_plot.to(device)

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)

    if args.track:
        wandb.init()
        # wandb.watch(model, log="all", log_freq=15)

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)

        for i, (img, label) in enumerate(progress_bar): # Remember, label is not used here

            examples_seen += img.size(0)

            img = img.to(device)
            img_reconstructed = model(img)

            loss = loss_fn(img, img_reconstructed)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}, Loss = {loss.item():>10.3f}")

            if args.track:
                wandb.log({"loss": loss.item()}, step=examples_seen)
            
            if (time.time() - t_last > args.seconds_between_image_logs) or (epoch == args.epochs - 1 and i == len(trainloader) - 1):
                t_last += args.seconds_between_image_logs
                with t.inference_mode():
                    if args.track:
                        arr = show_images(model, data_to_plot, return_arr=True)
                        images = wandb.Image(arr, caption="Top: original, Bottom: reconstructed")
                        wandb.log({"images": [images]}, step=examples_seen)
                    else:
                        show_images(model, data_to_plot, return_arr=False)

    wandb.run.save()
    wandb.finish()

    return model

# %%
if MAIN:
    args = AutoencoderArgs()
    args.latent_dim_size = 5
    # model = train_autoencoder(args)

# %%
if MAIN:
    # Choose number of interpolation points, and interpolation range
    n_points = 11
    interpolation_range = (-10, 10)

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, args.latent_dim_size), device=device)
    x = t.linspace(*interpolation_range, n_points)
    latent_dim_data[:, :, 0] = x.unsqueeze(0)
    latent_dim_data[:, :, 1] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims", title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(title_text="x0", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
        yaxis=dict(title_text="x1", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
    )
    fig.show()

    # def write_to_html(fig, filename):
    #     with open(f"{filename}.html", "w") as f:
    #         f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
    # write_to_html(fig, 'autoencoder_interpolation.html')

# %%

def make_scatter_plot(model, trainset, data_to_plot, n_examples=5000):
    trainloader = DataLoader(trainset, batch_size=64)
    df_list = []
    device = next(model.parameters()).device
    for img, label in trainloader:
        output = model.encoder(img.to(device)).detach().cpu().numpy()
        for label_single, output_single in zip(label, output):
            df_list.append({
                "x0": output_single[0],
                "x1": output_single[1],
                "label": str(label_single.item()),
            })
        if (n_examples is not None) and (len(df_list) >= n_examples):
            break
    df = pd.DataFrame(df_list).sort_values("label")
    fig = px.scatter(df, x="x0", y="x1", color="label", template="ggplot2")
    fig.update_layout(height=1000, width=1000)

    output_on_data_to_plot = model.encoder(data_to_plot.to(device)).detach().cpu().numpy()
    data_to_plot = (data_to_plot.numpy() * 0.3081) + 0.1307
    data_to_plot = (255 * data_to_plot).astype(np.uint8).squeeze()
    for i in range(10):
        img = Image.fromarray(data_to_plot[i]).convert("L")
        from IPython.display import display
        x = output_on_data_to_plot[i][0]
        y = output_on_data_to_plot[i][1]
        fig.add_layout_image(
            source=img,
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=2,
            sizey=2,
    )

    fig.show()

if MAIN:
    pass
    # make_scatter_plot(model, trainset, data_to_plot)
    # Result: better than I expected, density is pretty uniform and most of the space is utilised, 
    # although this is only a cross section of 2 dimensions so is a small subset of total space

# %%

# ============================================ VAEs ============================================

class VAE(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim_size*2),
            Rearrange("b (n latent_dim) -> n b latent_dim", n=2) # makes it easier to separate mu and sigma
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )
    def sample_latent_vector(self, x: t.Tensor) -> t.Tensor:
        mu, logsigma = self.encoder(x)
        sigma = t.exp(logsigma)
        # z = t.randn(self.latent_dim_size).to(device)
        z = mu + sigma * t.randn_like(mu)
        return z

    def forward(self, x: t.Tensor) -> t.Tensor:
        mu, logsigma = self.encoder(x)
        sigma = t.exp(logsigma)
        # z = t.randn(self.latent_dim_size).to(device)
        z = mu + sigma * t.randn_like(mu)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logsigma

if MAIN:
    model = VAE(latent_dim_size=5).to(device).train()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    print(torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0).to(device)))

# %%

def plot_loss(loss_fns_dict):
    df = pd.DataFrame(loss_fns_dict)
    fig = px.line(df, template="simple_white")
    fig.show()

@dataclass
class VAEArgs(AutoencoderArgs):
    weight_decay: float = 1e-5
    beta_kl: float = 0.1

def train_vae(args: VAEArgs):

    if args.track:
        wandb.init(config=args.__dict__)
        
    t_last = time.time()
    
    model = VAE(args.latent_dim_size).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    
    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    data_to_plot = args.data_to_plot.to(device)

    examples_seen = 0
    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)

        for i, (img, label) in enumerate(progress_bar):

            img = img.to(device)
            img_reconstructed, mu, logsigma = model(img)

            reconstruction_loss = loss_fn(img, img_reconstructed)
            kl_div_loss = ( 0.5 * (mu ** 2 + t.exp(2 * logsigma) - 1) - logsigma ).mean() * args.beta_kl

            loss = reconstruction_loss + kl_div_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            examples_seen += img.shape[0]
            vars_to_log = dict(
                reconstruction_loss = reconstruction_loss.item(),
                kl_div_loss = kl_div_loss.item(),
                mean = mu.mean(),
                std = t.exp(logsigma).mean(),
                total_loss = loss.item(),
            )
            if args.track:
                wandb.log(vars_to_log, step=examples_seen)


            desc = ", ".join([f"Epoch {epoch+1}", *[f"{k} = {v:>7.3f}" for k, v in vars_to_log.items()]])
            progress_bar.set_description(desc)

            if (time.time() - t_last > args.seconds_between_image_logs) or (epoch == args.epochs - 1 and i == len(trainloader) - 1):
                t_last += args.seconds_between_image_logs
                with t.inference_mode():
                    if args.track:
                        arr = show_images(model, data_to_plot, return_arr=True)
                        images = wandb.Image(arr, caption="Top: original, Bottom: reconstructed")
                        wandb.log({"images": [images]}, step=examples_seen)
                    else:
                        show_images(model, data_to_plot)
    
    if args.track:
        wandb.finish()
    return model

# %%

if MAIN:
    args = VAEArgs()
    args.latent_dim_size = 10
    args.epochs = 5
    model = train_vae(args)

# %%

if MAIN:
    # Choose number of interpolation points
    n_points = 11
    interpolation_range = (-1, 1)

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, args.latent_dim_size), device=device)
    x = t.linspace(*interpolation_range, n_points)
    latent_dim_data[:, :, 4] = x.unsqueeze(0)
    latent_dim_data[:, :, 3] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims", title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(title_text="x0", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
        yaxis=dict(title_text="x1", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
    )
    fig.show()

# %%

def make_3d_scatter_plot(model, trainset, n_examples=5000):
    trainloader = DataLoader(trainset, batch_size=64)
    df_list = []
    device = next(model.parameters()).device
    for img, label in trainloader:
        output = model.sample_latent_vector(img.to(device)).detach().cpu().numpy()
        for label_single, output_single in zip(label, output):
            df_list.append({
                "x0": output_single[0],
                "x1": output_single[1],
                "x2": output_single[2],
                "label": str(label_single.item()),
            })
        if (n_examples is not None) and (len(df_list) >= n_examples):
            break
    df = pd.DataFrame(df_list).sort_values("label")
    # create 3d scatter plot
    fig = px.scatter_3d(df, x="x0", y="x1", z="x2", color="label", template="ggplot2")
    fig.update_layout(height=1000, width=1000)
    fig.update_traces(marker_size = 4)
    fig.show()

if MAIN:
    make_3d_scatter_plot(model, args.trainset)
# %%
