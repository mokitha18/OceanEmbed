import numpy as np
import xarray as xr
import torch
import os
from .model import OceanEmbedCNN
from data_pipeline.harmonizer import prepare_training_dataset

def calculate_argo_metrics(predictions, argo_observations):
    """
    Compares model predictions against independent ARGO profiles.
    Computes depth-wise RMSE, Bias, and Pearson Correlation.
    """
    rmse = np.sqrt(np.mean((predictions - argo_observations) ** 2))
    bias = np.mean(predictions - argo_observations)
    
    if np.std(predictions) == 0 or np.std(argo_observations) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(predictions.flatten(), argo_observations.flatten())[0, 1]
    
    return rmse, bias, corr

if __name__ == "__main__":
    print("Loading ARGO observations...")
    argo_path = "data/raw/argo_validation.nc"
    if not os.path.exists(argo_path):
        print(f"File {argo_path} not found. Please run download_training.py")
        exit(1)
        
    argo_ds = xr.open_dataset(argo_path)
    argo_obs = argo_ds['thetao_argo'].values
    
    print("Loading model and generating predictions for ARGO validation...")
    model_path = "models/oceanembed.pth"
    model = OceanEmbedCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"Warning: {model_path} not found. Using untrained model for validation.")
    model.eval()

    X, _ = prepare_training_dataset("data/raw")
    
    with torch.no_grad():
        preds = model(X).numpy()
    
    # Broadcast or repeat preds if single timestep
    if preds.shape[0] == 1 and argo_obs.shape[0] > 1:
        preds = np.repeat(preds, argo_obs.shape[0], axis=0)
    elif argo_obs.shape[0] == 1 and preds.shape[0] > 1:
        argo_obs = np.repeat(argo_obs, preds.shape[0], axis=0)
    
    print("Computing metrics...")
    # Mask out missing values
    valid_mask = ~np.isnan(argo_obs) & (argo_obs != 0)
    
    pred_valid = preds[valid_mask]
    argo_valid = argo_obs[valid_mask]
    
    rmse, bias, corr = calculate_argo_metrics(pred_valid, argo_valid)
    
    print("\n--- ARGO Validation Results ---")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"Bias: {bias:.4f} °C")
    print(f"Pearson Correlation: {corr:.4f}")
    print("-------------------------------")
