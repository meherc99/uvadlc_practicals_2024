import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from globals import FGSM, PGD, ALPHA, EPSILON, NUM_ITER

def denormalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = batch.device
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).to(device)
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def fgsm_attack(image, data_grad, epsilon = 0.25):
    # Get the sign of the data gradient (element-wise)
    # Create the perturbed image, scaled by epsilon
    # Make sure values stay within valid range
    perturbed_image = image + epsilon * data_grad.sign()
    return perturbed_image

    
def fgsm_loss(model, criterion, inputs, labels, defense_args, return_preds = True):
    alpha = defense_args[ALPHA]
    epsilon = defense_args[EPSILON]
    inputs.requires_grad = True

    # Hint: the inputs are used in two different forward passes,
    # so you need to make sure those don't clash    
    orig_imgs = inputs.clone().detach().requires_grad_(True)
    adv_imgs = inputs.clone().detach().requires_grad_(True)
    
    # Calculate the loss for the original image
    preds = model(orig_imgs)
    orig_loss = criterion(preds, labels)

    orig_loss.backward(retain_graph=True)

    # Implement the FGSM attack
    adv_imgs = fgsm_attack(orig_imgs, orig_imgs.grad.data, epsilon)
    
    # Calculate the perturbation
    # Calculate the loss for the perturbed image
    adv_preds = model(adv_imgs.detach())
    adv_loss = criterion(adv_preds, labels)
    # adv_loss.backward()
    
    # Combine the two losses
    combined_loss = (1 - alpha)*adv_loss + alpha*orig_loss

    if return_preds:
        _, preds = torch.max(preds, 1)
        return combined_loss, preds
    else:
        return combined_loss


def pgd_attack(model, data, target, criterion, args):
    alpha = args[ALPHA]
    epsilon = args[EPSILON]
    num_iter = args[NUM_ITER]

    # Implement the PGD attack
    # Start with a copy of the data
    orig_data = data.clone().detach().to(data.device)
    adv_data = data.clone().detach().to(data.device).requires_grad_(True)
    # Then iteratively perturb the data in the direction of the gradient
    # Hint: to make sure to each time get a new detached copy of the data,
    # to avoid accumulating gradients from previous iterations
    # Hint: it can be useful to use toch.nograd()
   
    for _ in range(num_iter):
        adv_data.requires_grad_(True)
        preds = model(adv_data)        
        loss = criterion(preds, target)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_data = fgsm_attack(adv_data, adv_data.grad.data, alpha)
            
            # Make sure to clamp the perturbation to the epsilon ball around the original data
            perturbations = torch.clamp(adv_data - orig_data, min=-epsilon, max=epsilon)
            adv_data = torch.clamp(orig_data + perturbations, min=0, max=1)
    
    adv_data = adv_data.detach()   
    perturbed_data = adv_data
    return perturbed_data


def test_attack(model, test_loader, attack_function, attack_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # Very important for attack!
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 

        # If the initial prediction is wrong, don't attack
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        
        if attack_function == FGSM: 
            # Get the correct gradients wrt the data
            # Perturb the data using the FGSM attack
            # Re-classify the perturbed image
            loss.backward()
            data_denorm = denormalize(data)
            perturbed_data = fgsm_attack(data_denorm, data.grad.data, epsilon = attack_args[EPSILON]) 
        elif attack_function == PGD:
            # Get the perturbed data using the PGD attack
            # Re-classify the perturbed image
            perturbed_data = pgd_attack(model, data, target, criterion, attack_args)
        else:
            print(f"Unknown attack {attack_function}")

        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                original_data = data.squeeze().detach().cpu()
                adv_ex = perturbed_data.squeeze().detach().cpu()
                adv_examples.append( (init_pred.item(), 
                                      final_pred.item(),
                                      denormalize(original_data), 
                                      denormalize(adv_ex)) )

    # Calculate final accuracy
    final_acc = correct/float(len(test_loader))
    print(f"Attack {attack_function}, args: {attack_args}\nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples