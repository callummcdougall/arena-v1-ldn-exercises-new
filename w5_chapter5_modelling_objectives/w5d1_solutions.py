# %%
import torch as t
from typing import Tuple
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import torchinfo
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import time
from dataclasses import dataclass
import sys
from torchvision import transforms, datasets

p = r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured"
# Replace the line above with your own root directory
os.chdir(p)
sys.path.append(p)
sys.path.append(p + r"\w5_chapter5_modelling_objectives")

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# assert str(device) == "cuda:0"

import w5d1_utils
import w5d1_tests
from w0d2_chapter0_convolutions.solutions import pad1d, pad2d, conv1d_minimal, conv2d_minimal, Conv2d, Linear, ReLU, Pair, IntOrPair
from w0d3_chapter0_resnets.solutions import BatchNorm2d, Sequential

MAIN = __name__ == "__main__"

# %%

def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.
    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    """

    batch, in_channels, width = x.shape
    in_channels_2, out_channels, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"

    x_mod = pad1d(x, left=kernel_width-1, right=kernel_width-1, pad_value=0)
    weights_mod = rearrange(weights.flip(-1), "i o w -> o i w")

    return conv1d_minimal(x_mod, weights_mod)

if MAIN:
    w5d1_tests.test_conv_transpose1d_minimal(conv_transpose1d_minimal)

def fractional_stride_1d(x, stride: int = 1):
    '''Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.
    x: shape (batch, in_channels, width)
    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    batch, in_channels, width = x.shape
    width_new = width + (stride - 1) * (width - 1) # the RHS of this sum is the number of zeros we need to add between elements
    x_new_shape = (batch, in_channels, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = t.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride] = x
    
    return x_new

if MAIN:
    w5d1_tests.test_fractional_stride_1d(fractional_stride_1d)

def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.
    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    """

    batch, ic, width = x.shape
    ic_2, oc, kernel_width = weights.shape
    assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {weights.shape}."

    # Apply spacing
    x_spaced_out = fractional_stride_1d(x, stride)

    # Apply modification (which is controlled by the padding parameter)
    padding_amount = kernel_width - 1 - padding
    assert padding_amount >= 0, "total amount padded should be positive"
    x_mod = pad1d(x_spaced_out, left=padding_amount, right=padding_amount, pad_value=0)

    # Modify weights, then return the convolution
    weights_mod = rearrange(weights.flip(-1), "i o w -> o i w")

    return conv1d_minimal(x_mod, weights_mod)

if MAIN:
    w5d1_tests.test_conv_transpose1d(conv_transpose1d)

# %%

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def fractional_stride_2d(x, stride_h: int, stride_w: int):
    """
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    """
    batch, in_channels, height, width = x.shape
    width_new = width + (stride_w - 1) * (width - 1)
    height_new = height + (stride_h - 1) * (height - 1)
    x_new_shape = (batch, in_channels, height_new, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = t.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride_h, ::stride_w] = x
    
    return x_new

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv_transpose2d using bias=False
    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)
    Returns: shape (batch, out_channels, output_height, output_width)
    """

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    batch, ic, height, width = x.shape
    ic_2, oc, kernel_height, kernel_width = weights.shape
    assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {weights.shape}."

    # Apply spacing
    x_spaced_out = fractional_stride_2d(x, stride_h, stride_w)

    # Apply modification (which is controlled by the padding parameter)
    pad_h_actual = kernel_height - 1 - padding_h
    pad_w_actual = kernel_width - 1 - padding_w
    assert min(pad_h_actual, pad_w_actual) >= 0, "total amount padded should be positive"
    x_mod = pad2d(x_spaced_out, left=pad_w_actual, right=pad_w_actual, top=pad_h_actual, bottom=pad_h_actual, pad_value=0)

    # Modify weights
    weights_mod = rearrange(weights.flip(-1, -2), "i o h w -> o i h w")

    # Return the convolution
    return conv2d_minimal(x_mod, weights_mod)

if MAIN:
    w5d1_tests.test_conv_transpose2d(conv_transpose2d)

# %%

class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.
        Name your weight field `self.weight` for compatibility with the tests.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size = force_pair(kernel_size)
        sf = 1 / (self.out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = nn.Parameter(sf * (2 * t.rand(in_channels, out_channels, *kernel_size) - 1))

    def forward(self, x: t.Tensor) -> t.Tensor:

        return conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([
            f"{key}={getattr(self, key)}"
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        ])


class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (t.exp(x) - t.exp(-x)) / (t.exp(x) + t.exp(-x))

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.where(x > 0, x, self.negative_slope * x)
    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 1 / (1 + t.exp(-x))

if MAIN:
    w5d1_tests.test_ConvTranspose2d(ConvTranspose2d)
    w5d1_tests.test_Tanh(Tanh)
    w5d1_tests.test_LeakyReLU(LeakyReLU)
    w5d1_tests.test_Sigmoid(Sigmoid)

# %%

# Create a `Sequential` which can accept an ordered dict, and which can be iterated through
# (this is optional, not a required task)
# class Sequential(nn.Module):
#     def __init__(self, *modules: nn.Module):
#         super().__init__()
#         if isinstance(modules[0], OrderedDict):
#             for name, mod in modules[0].items():
#                 self.add_module(name, mod)
#         else:
#             for i, mod in enumerate(modules):
#                 self.add_module(str(i), mod)

#     def __getitem__(self, idx):
#         return list(self._modules.values())[idx]

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         """Chain each module together, with the output from one feeding into the next one."""
#         for mod in self._modules.values():
#             if mod is not None:
#                 x = mod(x)
#         return x

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
    ):
        super().__init__()

        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        self.img_channels = img_channels

        # Define the first layer, i.e. latent dim -> (1024, 4, 4) and reshape
        assert img_size % (2 ** n_layers) == 0
        first_height = img_size // (2 ** n_layers)
        first_size = generator_num_features * first_height * first_height
        self.project_and_reshape = Sequential(
            Linear(latent_dim_size, first_size, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_height),
            BatchNorm2d(generator_num_features),
            ReLU(),
        )

        # Get the list of parameters for feeding into the conv layers
        # note that the last out_channels is 3, for the colors of an RGB image
        in_channels_list = (generator_num_features / 2 ** t.arange(n_layers)).to(int).tolist()
        out_channels_list = in_channels_list[1:] + [self.img_channels,]

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (ci, co) in enumerate(zip(in_channels_list, out_channels_list)):
            conv_layer = [ConvTranspose2d(ci, co, 4, 2, 1), ReLU() if i < n_layers - 1 else Tanh()]
            if i < n_layers - 1:
                conv_layer.insert(1, BatchNorm2d(co))
            conv_layer_list.append(Sequential(*conv_layer))
        
        self.layers = Sequential(*conv_layer_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.project_and_reshape(x)
        x = self.layers(x)

        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
    ):
        super().__init__()

        self.img_size = img_size
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        self.img_channels = img_channels

        # Get the list of parameters for feeding into the conv layers
        # note that the last out_channels is 3, for the colors of an RGB image
        out_channels_list = (generator_num_features / 2 ** t.arange(n_layers)).to(int).tolist()[::-1]
        in_channels_list = [self.img_channels,] + out_channels_list

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (ci, co) in enumerate(zip(in_channels_list, out_channels_list)):
            conv_layer = [Conv2d(ci, co, 4, 2, 1), LeakyReLU(negative_slope = 0.2)]
            if i > 0:
                conv_layer.insert(1, BatchNorm2d(co))
            conv_layer_list.append(Sequential(*conv_layer))
        
        self.layers = Sequential(*conv_layer_list)

        # Define the last layer, i.e. reshape and (1024, 4, 4) -> real/fake classification
        assert img_size % (2 ** n_layers) == 0
        first_height = img_size // (2 ** n_layers)
        final_size = generator_num_features * first_height * first_height
        self.classifier = Sequential(
            Rearrange("b c h w -> b (c h w)"),
            Linear(final_size, 1, bias=False),
            Sigmoid()
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.layers(x)
        x = self.classifier(x)

        return x


def initialize_weights(model) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ConvTranspose2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, BatchNorm2d):
            nn.init.normal_(module.bias.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        generator_num_features: int = 1024,
        n_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.netD = Discriminator(img_size, img_channels, generator_num_features, n_layers)
        self.netG = Generator(latent_dim_size, img_size, img_channels, generator_num_features, n_layers)
        initialize_weights(self)

celeba_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 1024,
    n_layers = 4,
)
celeba_mini_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 512,
    n_layers = 4,
)

celeb_DCGAN = DCGAN(**celeba_config).to(device).train()
celeb_mini_DCGAN = DCGAN(**celeba_mini_config).to(device).train()

# %%

# ======================== CELEB_A ========================

if MAIN:
    image_size = 64

    transform = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.ImageFolder(
        root=r"celeba",
        transform=transform
    )

    w5d1_utils.show_images(trainset, rows=3, cols=5)

# ======================== MNIST ========================

# img_size = 24

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)


# %%

@dataclass
class DCGANargs():
    latent_dim_size: int
    img_size: int
    img_channels: int
    generator_num_features: int
    n_layers: int
    trainset: datasets.ImageFolder
    batch_size: int = 8
    epochs: int = 1
    lr: float = 0.0002
    betas: Tuple[float] = (0.5, 0.999)
    track: bool = True
    cuda: bool = True
    seconds_between_image_logs: int = 40

def train_DCGAN(args: DCGANargs) -> DCGAN:

    last_log_time = time.time()
    n_examples_seen = 0

    device = t.device("cuda" if args.cuda else "cpu")

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True) # num_workers=2

    model = DCGAN(
        args.latent_dim_size,
        args.img_size,
        args.img_channels,
        args.generator_num_features,
        args.n_layers,
    ).to(device).train()
    optG = t.optim.Adam(model.netG.parameters(), lr=args.lr, betas=args.betas)
    optD = t.optim.Adam(model.netD.parameters(), lr=args.lr, betas=args.betas)

    if args.track:
        wandb.init()
        wandb.watch(model)
    
    for epoch in range(args.epochs):
        
        progress_bar = tqdm(trainloader)

        for img_real, label in progress_bar: # remember that label is not used

            img_real = img_real.to(device)
            noise = t.randn(args.batch_size, model.netG.latent_dim_size).to(device)

            # ====== DISCRIMINIATOR TRAINING LOOP: maximise log(D(x)) + log(1-D(G(z))) ======

            # Zero gradients
            optD.zero_grad()
            # Calculate the two different components of the objective function
            D_x = model.netD(img_real)
            img_fake = model.netG(noise)
            D_G_z = model.netD(img_fake.detach())
            # Add them to get the objective function
            lossD = - (t.log(D_x).mean() + t.log(1 - D_G_z).mean())
            # Gradient descent step
            lossD.backward()
            optD.step()

            # ====== GENERATOR TRAINING LOOP: maximise log(D(G(z))) ======
            
            # Zero gradients
            optG.zero_grad()
            # Calculate the objective function
            D_G_z = model.netD(img_fake)
            lossG = - (t.log(D_G_z).mean())
            # Gradient descent step
            lossG.backward()
            optG.step()

            # Update progress bar
            progress_bar.set_description(f"{epoch=}, lossD={lossD.item():.4f}, lossG={lossG.item():.4f}")
            n_examples_seen += img_real.shape[0]

            # Log output, if required
            if args.track:
                wandb.log(dict(lossD=lossD, lossG=lossG), step=n_examples_seen)
                if time.time() - last_log_time > args.seconds_between_image_logs:
                    last_log_time = time.time()
                    arrays = get_generator_output(model.netG) # shape (8, 64, 64, 3)
                    images = [wandb.Image(arr) for arr in arrays]
                    wandb.log({"images": images}, step=n_examples_seen)

    name = model.__class__.__name__
    dirname = str(wandb.run.dir) if args.track else "models"
    filename = f"{dirname}/{name}.pt"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if args.track:
        print(f"Saving {name!r} to: {filename!r}")
        wandb.save(filename)
        wandb.finish()
                
    return model

@t.inference_mode()
def get_generator_output(netG, n_examples=8, rand_seed=0):
    netG.eval()
    device = next(netG.parameters()).device
    t.manual_seed(rand_seed)
    noise = t.randn(n_examples, netG.latent_dim_size).to(device)
    arrays = rearrange(netG(noise), "b c h w -> b h w c").detach().cpu().numpy()
    netG.train()
    return arrays

# %%

if MAIN:
    model = DCGAN(**celeba_config).to(device).train()
    # print_param_count(model)
    x = t.randn(3, 100).to(device)
    statsG = torchinfo.summary(model.netG, input_data=x)
    statsD = torchinfo.summary(model.netD, input_data=model.netG(x))
    print(statsG, statsD)

# %%

if MAIN:
    args = DCGANargs(**celeba_mini_config, trainset=trainset)

    model = train_DCGAN(args)
# %%