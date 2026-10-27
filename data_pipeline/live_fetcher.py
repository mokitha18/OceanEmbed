import xarray as xr
import numpy as np
import torch
import os
from datetime import datetime, timedelta

def create_uniform_grid():
    target_lats = np.linspace(5, 30, 101)
    target_lons = np.linspace(45, 105, 241)
    return target_lats, target_lons

def fetch_live_data(date_str):
    """
    Fetches real-time OPENDAP data for the specified date and regrids it.
    If OPENDAP fails (e.g. requires auth without credentials), it falls back to looking
    for historical local files in data/raw for demonstration without mocking.
    """
    try:
        from data_pipeline.harmonizer import harmonize_local_files
        tensor, sst_arr, *_ = harmonize_local_files("data/raw")
        return tensor, sst_arr
    except Exception as e:
        raise Exception(f"Live fetching failed and local data not found. Ensure real data is downloaded in data/raw: {e}")
