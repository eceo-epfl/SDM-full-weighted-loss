import torch
import numpy as np


def get_species_weights(y_train, species_weights_method):
    if species_weights_method == "inversely_proportional":
        species_weights = y_train.shape[0] / (y_train.sum(0) + 1e-5)
    elif species_weights_method == "inversely_proportional_clipped":
        species_weights = y_train.shape[0] / (y_train.sum(0) + 1e-5)
        species_weights = np.clip(species_weights, 0.05, 20)
    elif species_weights_method == "inversely_proportional_sqrt":
        species_weights = np.sqrt(y_train.shape[0] / (y_train.sum(0) + 1e-5))
    elif species_weights_method == "uniform":
        species_weights = 2 * np.ones(y_train.shape[1])
    elif species_weights_method == "inversely_proportional_not_normalized":
        species_weights = 1 / (y_train.sum(0) + 1e-5)
    else:
        raise ValueError("species_weights_method must be 'inversely_proportional', \
                        'inversely_proportional_clipped', 'inversely_proportional_sqrt', \
                        'uniform' or 'inversely_proportional_not_normalized'")
    return species_weights


def full_weighted_loss(pred_x, y, pred_bg, species_weights):
    
    batch_size = pred_x.size(0)
        
    # loss at data location
    loss_dl_pos = (log_loss(pred_x) * y * species_weights.repeat((batch_size, 1))).mean()
    loss_dl_neg = (log_loss(1 - pred_x) * (1 - y) * (species_weights/(species_weights - 1)).repeat((batch_size, 1))).mean()
    
    # loss at random location
    loss_rl = log_loss(1 - pred_bg).mean()
        
    return loss_dl_pos, loss_dl_neg, loss_rl


def log_loss(pred):
    """Helper function."""
    return -torch.log(pred + 1e-5)


### Baselines

def SSDL_loss(pred_x, y, pred_bg, species_weights):
    """From Cole et al. (2023)"""
    
    batch_size = pred_x.size(0)
        
    # loss at data location
    loss_dl_pos = (log_loss(pred_x) * y).sum()/y.sum()
    
    # loss at random location
    ran_species = np.random.randint(0, y.shape[1], size=(batch_size, 1))
    loss_rl = log_loss(1 - pred_bg)[np.arange(batch_size), np.ndarray.flatten(ran_species)].sum()/y.sum()
        
    return loss_dl_pos, torch.tensor(0), loss_rl


def SLDS_loss(pred_x, y, pred_bg, species_weights):
    """From Cole et al. (2023)"""
    
    batch_size = pred_x.size(0)
        
    # loss at data location
    loss_dl_pos = (log_loss(pred_x) * y).sum()/y.sum()
    ran_species = np.array([np.random.choice(row[y.cpu()[i] == 0], 1)[0] for i, row in enumerate(np.tile(np.arange(y.shape[1]), (batch_size, 1)))])
    loss_dl_neg = (log_loss(1 - pred_x) * (1 - y))[np.arange(batch_size), np.ndarray.flatten(ran_species)].sum()/y.sum()
        
    return loss_dl_pos, loss_dl_neg, torch.tensor(0)


def cat_ce_loss(pred_x, y, pred_bg, species_weights):
    """Categorical cross entropy loss"""
    
    pred = torch.logit(pred_x)
    pred = log_loss(torch.nn.functional.softmax(pred, dim=1))
    loss = (pred * y).mean()
    return loss, torch.tensor(0), torch.tensor(0)


def weighted_cat_ce_loss(pred_x, y, pred_bg, species_weights):
    """Weighted categorical cross entropy loss"""
    
    batch_size = pred_x.size(0)
    
    pred = torch.logit(pred_x)
    pred = log_loss(torch.nn.functional.softmax(pred, dim=1))
    loss = (pred * y * species_weights.repeat((batch_size, 1))).mean()
    return loss, torch.tensor(0), torch.tensor(0)


def focal_cat_ce_loss(pred_x, y, pred_bg, species_weights):
    """From Lin et al. (2017)"""
    
    gamma = 5 # 0.5, 1, 2, 5
    
    pred = torch.logit(pred_x)
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = ((1 - pred) ** gamma) * log_loss(pred)
    
    loss = (pred * y).mean()
    return loss, torch.tensor(0), torch.tensor(0)


def focal_loss(pred_x, y, pred_bg, species_weights):
    """Full weighted loss but with focal terms from Lin et al. (2017)"""
    
    batch_size = pred_x.size(0)
    gamma = 5  # 0.5, 1, 2, 5
    
    # loss at data location
    loss_dl_pos = (((1 - pred_x) ** gamma) * log_loss(pred_x) * y * species_weights.repeat((batch_size, 1))).mean()
    loss_dl_neg = (((pred_x ** gamma) * log_loss(1 - pred_x) * (1 - y) * (species_weights/(species_weights - 1)).repeat((batch_size, 1)))).mean()
    
    # loss at random location
    loss_rl = ((pred_bg ** gamma) * log_loss(1 - pred_bg)).mean()
        
    return loss_dl_pos, loss_dl_neg, loss_rl


def ldam_loss(pred_x, y, pred_bg, species_weights):
    """From Cao et al. (2019). Warning: Requires the species weights: inversely_proportional_not_normalized."""
    
    batch_size = pred_x.size(0)
    C = 0.1 # 0.1, 1, 10
    
    pred = torch.logit(pred_x)
    delta = (C * torch.sqrt(torch.sqrt(species_weights.repeat((batch_size, 1))))) * y
    pred = pred - delta
    pred = log_loss(torch.nn.functional.softmax(pred, dim=1))
    loss = (pred * y).mean()
    
    return loss, torch.tensor(0), torch.tensor(0)


def DB_loss(pred_x, y, pred_bg, species_weights):
    """From Wu et al. (2020). Warning: Requires the species weights: inversely_proportional_not_normalized."""
    
    batch_size = pred_x.size(0)
    lambda_ = 5 # 3, 5
    alpha = 0.1
    beta = 10
    mu = 0.2
    kappa = 0
    
    r = torch.mm((1/(y * species_weights.repeat((batch_size, 1))).sum(1)).unsqueeze(1), species_weights.unsqueeze(0))
    r_hat = alpha + 1/(1 + torch.exp(-beta * (r - mu)))
    pred = torch.logit(pred_x)
    loss = (r_hat * (y * torch.log(1 + torch.exp(-pred)) + (1/lambda_)* (1 - y) * torch.log(1 + torch.exp(lambda_ * pred)))).mean()
    
    return loss, torch.tensor(0), torch.tensor(0)


def entmax_loss(pred_x, y, pred_bg, species_weights):
    """From Zhou et al. (2022)"""
    
    batch_size = pred_x.size(0)
    alpha = 0.01 # 0.01, 0.05, 0.1
        
    # loss at data location
    loss_dl_pos = (log_loss(pred_x) * y).mean()
    loss_dl_neg = -alpha * (pred_x * log_loss(pred_x) * (1 - y) + (1 - pred_x) * log_loss(1 - pred_x) * (1 - y)).mean()
    
    # loss at random location
    loss_rl = -alpha *((pred_bg) * log_loss(pred_bg) + (1 - pred_bg) * log_loss(1 - pred_bg)).mean()
        
    return loss_dl_pos, loss_dl_neg, loss_rl
