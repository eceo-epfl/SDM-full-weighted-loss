import os
import random
import numpy as np
import torch

from data_helpers import SpeciesDataset
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from models import MLP
from losses import full_weighted_loss, get_species_weights
from prg_corrected import create_prg_curve, calc_auprg


def seed_everything(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def train_multi_species(x_train, y_train, bg_train, x_val, y_val, bg_val, x_test, y_test, config):
    """Train a multi-species model."""
    
    # We compute the AUC only for species represented in both train and val sets
    if config["cross_validation"]:
        indices_non_zeros_samples = np.intersect1d(np.sum(y_val, axis=0).nonzero(), 
                                                    np.sum(y_train, axis=0).nonzero())
    else:
        indices_non_zeros_samples = np.full(y_train.shape[1], True)
    
    # Species weights
    species_weights = get_species_weights(y_train, config["species_weights_method"])
    species_weights = torch.tensor(species_weights, dtype=torch.float32).to(config["device"])
    
    # DataLoader
    x_train = torch.tensor(x_train, dtype=torch.float32).to(config["device"])
    y_train = torch.tensor(y_train, dtype=torch.float32).to(config["device"])
    bg_train = torch.tensor(bg_train, dtype=torch.float32).to(config["device"])
    if config["cross_validation"]:
        x_val = torch.tensor(x_val, dtype=torch.float32).to(config["device"])
        y_val = torch.tensor(y_val, dtype=torch.float32).to(config["device"])
        bg_val = torch.tensor(bg_val, dtype=torch.float32).to(config["device"])
    trainset = SpeciesDataset(x_train, y_train, bg_train)
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    
    # Model and optimizer
    model = MLP(input_size=x_train.shape[1], output_size=config["num_species"], num_layers=config["num_layers"], 
                width=config["width_MLP"], dropout=config["dropout"]).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_lambda = lambda epoch: config["learning_rate_decay"] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    loss_fn = config["loss_fn"]
    for i in range(config["epochs"]):

        if i > 0: 
            scheduler.step()

        for x_batch, y_batch, bg_batch in trainloader:

            # Forward pass
            output = torch.sigmoid(model(torch.cat((x_batch, bg_batch), 0)))
            pred_x = output[:len(x_batch)]
            pred_bg = output[len(x_batch):]

            loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_x, y_batch, pred_bg, species_weights)
            
            # Compute total loss
            train_loss = config["lambda_1"] * loss_dl_pos + config["lambda_2"] * loss_dl_neg + (1 - config["lambda_2"]) * loss_rl
            
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            
            model.eval()
                        
            train_results = compute_metrics(model, x_train, y_train, bg_train, config, loss_fn, species_weights, indices_non_zeros_samples)
                    
            if config["cross_validation"] is not None:
                val_results = compute_metrics(model, x_val, y_val, bg_val, config, loss_fn, species_weights, indices_non_zeros_samples)
                print(f"Epoch {i} - train loss: {train_results['loss']:.4f} - val loss: {val_results['loss']:.4f} - val auc: {val_results['mean_auc_roc']:.4f}")
            else:
                print(f"Epoch {i} - train loss: {train_results['loss']:.4f}")
            
            model.train()
    
    mean_test_auc_roc, mean_test_auc_prg, mean_test_cor, mean_val_auc_roc = evaluate_model(model, 
                    x_val, y_val, bg_val, x_test, y_test, config, loss_fn, species_weights, indices_non_zeros_samples)
    
    return model, mean_test_auc_roc, mean_test_auc_prg, mean_test_cor, mean_val_auc_roc


def evaluate_model(model, x_val, y_val, bg_val, x_test, y_test, config, loss_fn, species_weights, indices_non_zeros_samples):
    """Evaluate the model on the validation and test set."""
                
    model.eval()
    
    # Validation set
    if config["cross_validation"] is not None:
        val_results = compute_metrics(model, x_val, y_val, bg_val, config, loss_fn, species_weights, indices_non_zeros_samples)
        mean_val_auc_roc = val_results["mean_auc_roc"]
    else:
        mean_val_auc_roc = None

    # Test set
    test_pred = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32).to(config["device"]))).cpu().detach().numpy()
    non_nan_elements = np.logical_not(np.isnan(y_test)) # NaN elements correspond to other groups of species 
    test_auc_rocs = []
    test_auc_prgs = []
    test_cors = []
    for j in range(y_test.shape[1]):
        y_test_col = y_test[:, j]
        y_test_col = y_test_col[non_nan_elements[:, j]]
        test_pred_col = test_pred[:, j]
        test_pred_col = test_pred_col[non_nan_elements[:, j]]        
        test_auc_rocs.append(roc_auc_score(y_test_col, test_pred_col))
        prg_curve = create_prg_curve(y_test_col, test_pred_col)
        test_auc_prgs.append(calc_auprg(prg_curve))
        test_cors.append(pearsonr(y_test_col, test_pred_col).statistic)
    mean_test_auc_roc = np.mean(test_auc_rocs)
    mean_test_auc_prg = np.mean(test_auc_prgs)
    mean_test_cor = np.mean(test_cors)
    
    return mean_test_auc_roc, mean_test_auc_prg, mean_test_cor, mean_val_auc_roc


def compute_metrics(model, x, y, bg, config, loss_fn, species_weights, indices_non_zeros_samples):
    """Compute the different metrics of the model."""
    
    pred = torch.sigmoid(model(torch.cat((x, bg[:len(x)]), 0)))
    pred_x = pred[:len(x)]
    pred_bg = pred[len(x):]
    
    # Loss function in pytorch
    loss_dl_pos, loss_dl_neg, loss_rl = loss_fn(pred_x, y, pred_bg, species_weights)
    loss_dl_pos, loss_dl_neg, loss_rl = loss_dl_pos.item(), loss_dl_neg.item(), loss_rl.item()
    if len(pred_bg) == 0:
        loss_rl = 0
    loss = config["lambda_1"] * loss_dl_pos + config["lambda_2"] * loss_dl_neg + (1 - config["lambda_2"]) * loss_rl
    
    # Metrics in numpy
    pred = pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    
    y_with_bg = np.concatenate((y, np.zeros((len(pred_bg), y.shape[1]))), axis=0)
        
    auc_rocs = list(roc_auc_score(y_with_bg[:, indices_non_zeros_samples], pred[:, indices_non_zeros_samples], average=None))
    auc_prgs = []
    cors = []
    for j in range(len(y_with_bg[0])):
        if j in indices_non_zeros_samples:
            prg_curve = create_prg_curve(y_with_bg[:, j], pred[:, j])
            auc_prgs.append(calc_auprg(prg_curve))
            cors.append(pearsonr(y_with_bg[:, j], pred[:, j]).statistic)
    
    mean_auc_roc = np.mean(auc_rocs)
    mean_auc_prg = np.mean(auc_prgs)
    mean_cor = np.mean(cors)

    results = {
        "loss_dl_pos": loss_dl_pos,
        "loss_dl_neg": loss_dl_neg,
        "loss_rl": loss_rl,
        "loss": loss,
        "auc_rocs": auc_rocs,
        "auc_prgs": auc_prgs,
        "cors": cors,
        "mean_auc_roc": mean_auc_roc,
        "mean_auc_prg": mean_auc_prg,
        "mean_cor": mean_cor
    }

    return results
