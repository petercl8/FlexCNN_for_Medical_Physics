import torch

def get_supervisory_loss(fake_X, real_X, sup_base_criterion):
    '''
    Function to calculate the supervisory loss.

    fake_X:         fake image tensor (Terminology from GANs. For supervisory networks, it's arbitrary whether fake_X or real_X are ground truths or reconstructions)
    real_X:         real image tensor
    sup_base_criterion   loss function. Will be a Pytorch object.
    '''
    #print('Calc supervisory loss')
    sup_loss = sup_base_criterion(fake_X, real_X)
    return sup_loss

def get_disc_loss(fake_X, real_X, disc_X, adv_criterion):
    '''
    Function to calculate the discriminator loss. Used to train the discriminator.
    '''
    disc_fake_pred = disc_X(fake_X.detach()) # Detach generator from fake batch
    disc_fake_loss = adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)) # Good fakes shoudl yield predictions = 0.
    disc_real_pred = disc_X(real_X)
    disc_real_loss = adv_criterion(disc_real_pred, torch.ones_like(disc_real_pred)) # Good fakes shoudl yield predictions = 1.
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_gen_adversarial_loss(real_X, gen_XY, disc_Y, adv_criterion):
    '''
    Function to calculate the adversarial loss (for gen_XY) and fake_Y (from real_X).
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) # generator is penalized for discriminmator getting it right
    return adversarial_loss, fake_Y

def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Function to calculate the cycle-consistency loss (for gen_YX).
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    return cycle_loss, cycle_X

def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, config):
    '''
    Function to calculate the total generator loss. Used to train the generators.
    
    Returns:
        gen_loss: Total weighted generator loss tensor.
        adv_loss: Adversarial loss scalar (0 if lambda_adv=0).
        sup_loss: Supervisory loss scalar (0 if lambda_sup=0).
        cycle_loss: Cycle-consistency loss scalar (0 if lambda_cycle=0 or cycle_criterion=None).
        cycle_A: Reconstructed A tensor via cycle consistency (None if cycle disabled).
        cycle_B: Reconstructed B tensor via cycle consistency (None if cycle disabled).
    '''
    supervisory_criterion = config['sup_base_criterion']
    cycle_criterion = config['cycle_criterion']
    gen_adversarial_criterion = config['gen_adv_criterion']
    lambda_adv = config['lambda_adv']
    lambda_sup = config['lambda_sup']
    lambda_cycle = config['lambda_cycle']

    # Adversarial Loss
    if lambda_adv != 0: # To save resources, we only run this code if lambda_adv != 0
        adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, gen_AB, disc_B, gen_adversarial_criterion)
        adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, gen_BA, disc_A, gen_adversarial_criterion)
        adv_loss = adv_loss_AB+adv_loss_BA
    else: # Even if we don't compute adversarial losses, we still need fake_A and fake_B for later code
        fake_A = gen_BA(real_B)
        fake_B = gen_AB(real_A)
        adv_loss = 0

    # Supervisory Loss
    if lambda_sup != 0: # To save resources, we only run this code if lambda_sup != 0
        sup_loss_AB = get_supervisory_loss(fake_B, real_B, supervisory_criterion)
        sup_loss_BA = get_supervisory_loss(fake_A, real_A, supervisory_criterion)
        sup_loss = sup_loss_AB+sup_loss_BA
    else:
        sup_loss = 0

    # Cycle-consistency Loss (gated by both lambda_cycle and criterion availability)
    if lambda_cycle != 0 and cycle_criterion is not None:
        cycle_loss_AB, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
        cycle_loss_BA, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss = cycle_loss_AB+cycle_loss_BA
    else:
        cycle_loss = 0
        cycle_A = None
        cycle_B = None

    # Total Generator Loss
    gen_loss = lambda_adv*adv_loss+lambda_sup*sup_loss+lambda_cycle*cycle_loss
    
    # Extract scalar loss values for return (monitoring purposes)
    adv_loss_scalar = adv_loss.item() if lambda_adv != 0 else 0
    sup_loss_scalar = sup_loss.item() if lambda_sup != 0 else 0
    cycle_loss_scalar = cycle_loss.item() if lambda_cycle != 0 and cycle_criterion is not None else 0
    
    return gen_loss, adv_loss_scalar, sup_loss_scalar, cycle_loss_scalar, cycle_A, cycle_B


def patchwise_moment_loss(batch_pred,
                          batch_target,
                          moments=[1,2],                    # 1=mean, 2=std
                          moment_weights={1:2.0, 2:1.0},    # relative importance of each moment
                          patch_size=8,
                          stride=4,
                          eps=1e-6,
                          patch_weighting='scaled',          # 'scaled', 'energy', 'mean', 'none'
                          patch_weight_min=0.33,
                          patch_weight_max=1.0,
                          min_patch_mean=1e-3):
    """
    Fully vectorized patchwise moment loss (mean + std) suitable for GPU backpropagation.

    Features:
    - Physics-informed normalization: mean -> μ, std -> √μ
    - Patch weighting retained
    - Low-count patch masking
    - Fully vectorized for speed in training
    """

    B, C, H, W = batch_pred.shape
    p, s = patch_size, stride

    # Only full patches
    num_patches_h = (H - p) // s + 1
    num_patches_w = (W - p) // s + 1
    if num_patches_h <= 0 or num_patches_w <= 0:
        raise ValueError("Patch size larger than image dimensions.")
    max_h = s * (num_patches_h - 1) + p
    max_w = s * (num_patches_w - 1) + p
    batch_pred = batch_pred[:, :, :max_h, :max_w]
    batch_target = batch_target[:, :, :max_h, :max_w]

    # Extract patches and flatten spatial dims
    pred_patches = batch_pred.unfold(2, p, s).unfold(3, p, s)
    target_patches = batch_target.unfold(2, p, s).unfold(3, p, s)
    num_patches = num_patches_h * num_patches_w
    pred_patches = pred_patches.contiguous().view(B, C, num_patches, -1)
    target_patches = target_patches.contiguous().view(B, C, num_patches, -1)

    # Patch mean (shape [B, C, num_patches])
    patch_mean = target_patches.mean(dim=-1)

    # Patch weighting
    patch_min = patch_mean.min(dim=-1, keepdim=True)[0]
    patch_max = patch_mean.max(dim=-1, keepdim=True)[0]

    if patch_weighting == 'scaled':
        patch_weights = patch_weight_min + \
                        (patch_mean - patch_min) / (patch_max - patch_min + eps) * \
                        (patch_weight_max - patch_weight_min)
    elif patch_weighting == 'energy':
        patch_energy = (target_patches ** 2).mean(dim=-1)
        patch_weights = patch_energy / (patch_energy.sum(dim=-1, keepdim=True) + eps)
    elif patch_weighting == 'mean':
        patch_weights = patch_mean / (patch_mean.sum(dim=-1, keepdim=True) + eps)
    else:
        patch_weights = torch.ones_like(patch_mean)

    # Low-count patch mask
    patch_mask = (patch_mean >= min_patch_mean).float()
    patch_weights = patch_weights * patch_mask

    # Compute all patch statistics at once
    target_mean = patch_mean
    pred_mean = pred_patches.mean(dim=-1)

    target_centered = target_patches - target_mean.unsqueeze(-1)
    pred_centered = pred_patches - pred_mean.unsqueeze(-1)

    target_std = torch.sqrt((target_centered ** 2).mean(dim=-1) + eps)
    pred_std = torch.sqrt((pred_centered ** 2).mean(dim=-1) + eps)

    rel_diff_dict = {}

    total_metric = 0.0
    for k in moments:
        if k == 1:
            # Mean differences
            rel_diff = torch.abs(pred_mean - target_mean) / (target_mean + eps)
            denom_factor = moment_weights.get(k, 1.0)
        elif k == 2:
            # Std differences normalized by sqrt(mean)
            rel_diff = torch.abs(pred_std - target_std) / (torch.sqrt(target_mean + eps))
            denom_factor = moment_weights.get(k, 1.0)
        else:
            raise ValueError("Only mean (1) and std (2) supported.")

        # Weighted aggregation over patches, channels, batch
        weighted_diff = (rel_diff * patch_weights).sum(dim=-1).mean(dim=[0,1])
        weighted_diff *= denom_factor

        rel_diff_dict[k] = weighted_diff
        total_metric += weighted_diff

    # Normalize by sum of moment weights
    total_metric /= sum(moment_weights.get(k,1.0) for k in moments)

    return total_metric


### Functons for Assymmetric/Separate (Older) ###
'''
def get_gen_adv_loss(fake_X, disc_X, adv_criterion):
    print('Calc generative adversarial loss')
    disc_fake_pred = disc_X(fake_X)
    adversarial_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) # Called only from get_gen_loss
    return adversarial_loss

def get_sup_loss(fake_X, real_X, sup_base_criterion):
    print('Calc supervisory loss')
    sup_loss = sup_base_criterion(fake_X, real_X)
    return sup_loss

def get_cycle_loss(fake_I, gen_IS, low_rez_S, cycle_criterion):
    print('Calc cycle loss')
    cycle_S = gen_IS(fake_I)
    cycle_loss = cycle_criterion(cycle_S, low_rez_S)
'''