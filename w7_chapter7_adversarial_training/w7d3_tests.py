# %%
import numpy as np
import torch
import torch.nn.functional as F
from w7_chapter7_adversarial_training import w7d3_utils
from torchvision import models
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def test_untargeted_attack(untargeted_adv_attack, eps=0.01):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, true_index = w7d3_utils.load_example_image(preprocess=False)
    norm_image = w7d3_utils.IMAGENET_NORMALIZE(image)

    # Generate predictions
    _, index, confidence = w7d3_utils.make_single_prediction(model, norm_image)
    label = w7d3_utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Generate Adversarial Example
    true_index = torch.Tensor([true_index]).type(torch.long)
    adv_image = untargeted_adv_attack(
        image.unsqueeze(0), 
        true_index, 
        model, 
        w7d3_utils.IMAGENET_NORMALIZE, 
        eps=eps
    ).squeeze(0)
    norm_adv_image = w7d3_utils.IMAGENET_NORMALIZE(adv_image.squeeze(0))

    # Display Results
    _, adv_index, adv_confidence = w7d3_utils.make_single_prediction(model, norm_adv_image)
    adv_label = w7d3_utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    w7d3_utils.display_adv_images(
        image, 
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )


def test_targeted_attack(targeted_adv_attack, target_idx=10, eps=0.01):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, _ = w7d3_utils.load_example_image(preprocess=False)
    norm_image = w7d3_utils.IMAGENET_NORMALIZE(image)
    
    # Generate predictions
    _, index, confidence = w7d3_utils.make_single_prediction(model, norm_image)
    label = w7d3_utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = w7d3_utils.get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        w7d3_utils.IMAGENET_NORMALIZE, 
        eps=eps
    ).squeeze(0)
    norm_adv_image = w7d3_utils.IMAGENET_NORMALIZE(adv_image.squeeze(0))

    # Display Results
    _, adv_index, adv_confidence = w7d3_utils.make_single_prediction(model, norm_adv_image)
    adv_label = w7d3_utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    w7d3_utils.display_adv_images(
        image,
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )




