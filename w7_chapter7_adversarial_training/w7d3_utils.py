import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from robustness import datasets, model_utils

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
IMAGENET_DENORMALIZE = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
])
torch.set_default_tensor_type(torch.cuda.FloatTensor)

with open('imagenet_labels.json', 'r') as f:
    IMAGENET_LABELS = json.load(f)


def load_example_image(preprocess=True):
    """Loads an example image of a Panda and its label.

    Args:
        preprocess (bool): If true, applies imagenet preprocessing.
    """
    image = np.load('images/panda.npy')
    image = torch.Tensor(image)
    label = 388
    image = image / 255
    if preprocess:
        image = IMAGENET_NORMALIZE(image)
    return image, label


def make_single_prediction(model, image):
    """Makes a prediction on a single image

    Args:
        model (nn.Module): the prediction network
        image (torch.Tensor): the input image
    """
    pred = model.forward(image.unsqueeze(0))
    pred = F.softmax(pred, dim=1)[0]
    pred_index = pred.argmax().item()
    pred_confidence = pred[pred_index]
    return pred, pred_index, pred_confidence


def make_batch_prediction(model, image_batch):
    """Makes predictions on a batch of images

    Args:
        model (nn.Module): the prediction network
        image_batch (torch.Tensor): the batch of images
    """
    pred = model.forward(image_batch)
    pred = F.softmax(pred, dim=1)
    pred_indices = pred.argmax(dim=1).item()
    pred_confidences = pred[range(pred.shape[0]), pred_indices]
    return pred, pred_indices, pred_confidences


def get_imagenet_label(index):
    """Gets the imagenet label of certain index

    Args:
        index (int): the index of the label, between 0 and 999, inclusive.
    """
    label = IMAGENET_LABELS[str(index)]
    return label


def display_adv_images(image, adv_image, pred, adv_pred, channels_first=False, denormalize=False):
    """Displays the normal and adversarial image side by side
    
    Args:
        image (torch.Tensor): The regular image
        adv_image (torch.Tensor): The adversarial image
        pred (float): A tuple containing (label, confidence) of the image
        adv_pred (float): A tuple containing (label, adv_confidence) of the adv_image
        channels_first (bool): Whether the images are channels first.
        denormalize (bool): Whether we should denormalize the image.
    """
    label, confidence = pred
    adv_label, adv_confidence = adv_pred
    if denormalize:
        image = IMAGENET_DENORMALIZE(image).detach().cpu().numpy()
        adv_image = IMAGENET_DENORMALIZE(adv_image).detach().cpu().numpy()
    else:
        image = image.detach().cpu().numpy()
        adv_image = adv_image.detach().cpu().numpy()
    if channels_first:
        image = np.moveaxis(image, 0, 2)
        adv_image = np.moveaxis(adv_image, 0, 2)
    image = image.clip(0,1)
    adv_image = adv_image.clip(0,1)
    _, axes = plt.subplots(1,2)
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].set(xlabel=f'Prediction: {label}\nConfidence: {confidence:.4f}')
    axes[1].imshow(adv_image)
    axes[1].set_title('Adversarial Image')
    axes[1].set(xlabel=f'Prediction: {adv_label}\nConfidence: {adv_confidence:.4f}')
    plt.show()

def attack_normal_model(
    targeted_adv_attack, 
    target_idx=10, 
    eps=0.03, 
    num_steps=10, 
    step_size=0.01
):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, _ = load_example_image(preprocess=False)
    norm_image = IMAGENET_NORMALIZE(image)

    # Generate predictions
    _, index, confidence = make_single_prediction(model, norm_image)
    label = get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        IMAGENET_NORMALIZE,
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)
    norm_adv_image = IMAGENET_NORMALIZE(adv_image.squeeze(0))

    # Display Results
    _, adv_index, adv_confidence = make_single_prediction(model, norm_adv_image)
    adv_label = get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    display_adv_images(
        image,
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )


def get_adv_trained_model():
    attack_model, _ = model_utils.make_and_restore_model(
        arch='resnet18', 
        dataset=datasets.ImageNet(''), 
        resume_path='checkpoints/resnet18_linf_eps8.0.ckpt'
    )
    model = attack_model.model
    return model


def attack_adversarially_trained_model(
    targeted_adv_attack, 
    target_idx=10, 
    eps=0.03, 
    num_steps=10, 
    step_size=0.01
):
    # Load the models
    model = get_adv_trained_model()
    print('')
    
    # Load the preprocessed image
    image, _ = load_example_image(preprocess=False)
    norm_image = IMAGENET_NORMALIZE(image)

    # Generate predictions
    _, index, confidence = make_single_prediction(model, norm_image)
    label = get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        IMAGENET_NORMALIZE,
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)
    norm_adv_image = IMAGENET_NORMALIZE(adv_image.squeeze(0))

    # Display Results
    _, adv_index, adv_confidence = make_single_prediction(model, norm_adv_image)
    adv_label = get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    display_adv_images(
        image,
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )