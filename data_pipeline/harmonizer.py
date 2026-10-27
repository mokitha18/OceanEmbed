import xarray as xr
import numpy as np
import torch
import os

def create_uniform_grid():
    """
    Creates the target 0.25 degree grid for the North Indian Ocean.
    Lat: 5N to 30N (101 points)
    Lon: 45E to 105E (241 points)
    """
    target_lats = np.linspace(5, 30, 101)
    target_lons = np.linspace(45, 105, 241)
    return target_lats, target_lons

def _get_var_coord_names(var):
    lat_name = None
    lon_name = None
    for d in var.dims:
        if d.lower() in ['lat', 'latitude', 'nav_lat', 'y']:
            lat_name = d
        elif d.lower() in ['lon', 'longitude', 'nav_lon', 'x']:
            lon_name = d
    if not lat_name:
        lat_name = 'lat' if 'lat' in var.coords else 'latitude'
    if not lon_name:
        lon_name = 'lon' if 'lon' in var.coords else 'longitude'
    return lat_name, lon_name

def _get_var(ds, candidate_names):
    for name in candidate_names:
        if name in ds.variables or name in ds.data_vars:
            return ds[name]
    raise KeyError(f"None of {candidate_names} found in dataset variables: {list(ds.variables.keys())}")

def _regrid_2d(var, target_lats, target_lons):
    lat_k, lon_k = _get_var_coord_names(var)
    # Fast path: check if coordinates already match target grid
    if (lat_k in var.coords and lon_k in var.coords and 
        len(var[lat_k]) == len(target_lats) and len(var[lon_k]) == len(target_lons) and
        np.allclose(var[lat_k].values, target_lats, atol=1e-3) and np.allclose(var[lon_k].values, target_lons, atol=1e-3)):
        interp_da = var
    else:
        interp_da = var.interp({lat_k: target_lats, lon_k: target_lons}, method='linear')
        
    if 'time' in interp_da.dims:
        interp_da = interp_da.isel(time=0)
    interp_da = interp_da.transpose(lat_k, lon_k)
    return interp_da.values

def harmonize_local_files(data_dir="data/raw"):
    """
    Reads SST, SSS, SSH, Currents, and Winds from local NetCDF files.
    Supports NASA PO.DAAC, Copernicus, and ERA5 formats.
    Regrids to a uniform 0.25 deg grid (101 lat, 241 lon).
    Returns PyTorch tensor: (Batch=1, Channels=5, H=101, W=241)
    """
    target_lats, target_lons = create_uniform_grid()
    
    sst_path = os.path.join(data_dir, "sst.nc")
    sss_path = os.path.join(data_dir, "sss.nc")
    ssh_path = os.path.join(data_dir, "ssh.nc")
    curr_path = os.path.join(data_dir, "currents.nc")
    wind_path = os.path.join(data_dir, "winds.nc")
    
    if not all(os.path.exists(p) for p in [sst_path, sss_path, ssh_path, curr_path, wind_path]):
        raise FileNotFoundError(f"One or more NetCDF surface files are missing from {data_dir}/ (needed: sst.nc, sss.nc, ssh.nc, currents.nc, winds.nc).")

    with xr.open_dataset(sst_path) as sst_ds:
        var_sst = _get_var(sst_ds, ['analysed_sst', 'sst', 'thetao'])
        sst_regrid = _regrid_2d(var_sst, target_lats, target_lons)
        if np.nanmean(sst_regrid) > 100.0:
            sst_regrid = sst_regrid - 273.15

    with xr.open_dataset(sss_path) as sss_ds:
        var_sss = _get_var(sss_ds, ['sos', 'sss', 'so'])
        sss_regrid = _regrid_2d(var_sss, target_lats, target_lons)

    with xr.open_dataset(ssh_path) as ssh_ds:
        var_ssh = _get_var(ssh_ds, ['sla', 'adt', 'zos', 'ssh'])
        ssh_regrid = _regrid_2d(var_ssh, target_lats, target_lons)

    with xr.open_dataset(curr_path) as curr_ds:
        var_u = _get_var(curr_ds, ['u', 'uo', 'u_curr', 'eastward_eulerian_current_velocity'])
        var_v = _get_var(curr_ds, ['v', 'vo', 'v_curr', 'northward_eulerian_current_velocity'])
        u_curr = _regrid_2d(var_u, target_lats, target_lons)
        v_curr = _regrid_2d(var_v, target_lats, target_lons)
        speed_regrid = np.sqrt(u_curr**2 + v_curr**2)

    with xr.open_dataset(wind_path) as wind_ds:
        var_uw = _get_var(wind_ds, ['uwnd', 'eastward_wind', 'u10', '10m_u_component_of_wind', 'u', 'wind_speed_u'])
        var_vw = _get_var(wind_ds, ['vwnd', 'northward_wind', 'v10', '10m_v_component_of_wind', 'v', 'wind_speed_v'])
        u_wind = _regrid_2d(var_uw, target_lats, target_lons)
        v_wind = _regrid_2d(var_vw, target_lats, target_lons)
        wind_speed_regrid = np.sqrt(u_wind**2 + v_wind**2)

    # Stack along channels: (5, 101, 241)
    tensor_data = np.stack([sst_regrid, sss_regrid, ssh_regrid, speed_regrid, wind_speed_regrid], axis=0)
    tensor_data = np.nan_to_num(tensor_data, nan=0.0)
    
    # Add batch dimension: (1, 5, 101, 241)
    input_tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)
    sst_array = np.nan_to_num(sst_regrid.copy(), nan=0.0)
    land_mask = np.isnan(sst_regrid)
    
    return input_tensor, sst_array, land_mask

def regrid_glorys_target(data_dir="data/raw"):
    """
    Regrids GLORYS subsurface target to the 15 standard depth levels and (101, 241) spatial grid.
    Returns: PyTorch tensor (Batch, Depths=15, H=101, W=241)
    """
    target_lats, target_lons = create_uniform_grid()
    standard_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    
    glorys_path = os.path.join(data_dir, "glorys_target.nc")
    if not os.path.exists(glorys_path):
        raise FileNotFoundError(f"glorys_target.nc is missing from {data_dir}/")
        
    with xr.open_dataset(glorys_path) as glorys_ds:
        depth_name = 'depth' if 'depth' in glorys_ds.coords else 'lev'
        var_target = _get_var(glorys_ds, ['thetao', 'temperature', 'temp'])
        lat_k, lon_k = _get_var_coord_names(var_target)
        
        # Check if coordinates already match target grid
        if (depth_name in var_target.coords and lat_k in var_target.coords and lon_k in var_target.coords and
            len(var_target[depth_name]) == len(standard_depths) and
            len(var_target[lat_k]) == len(target_lats) and
            len(var_target[lon_k]) == len(target_lons) and
            np.allclose(var_target[depth_name].values, standard_depths, atol=1e-2) and
            np.allclose(var_target[lat_k].values, target_lats, atol=1e-3) and
            np.allclose(var_target[lon_k].values, target_lons, atol=1e-3)):
            regrid = var_target
        else:
            regrid = var_target.interp({depth_name: standard_depths, lat_k: target_lats, lon_k: target_lons}, method='linear')
            
        if 'time' in regrid.dims:
            regrid = regrid.transpose('time', depth_name, lat_k, lon_k)
            val = regrid.values
        else:
            regrid = regrid.transpose(depth_name, lat_k, lon_k)
            val = regrid.values[np.newaxis, ...]
            
        if np.nanmean(val) > 100.0:
            val = val - 273.15
            
    val = np.nan_to_num(val, nan=0.0)
    return torch.tensor(val, dtype=torch.float32)

def prepare_training_dataset(data_dir="data/raw"):
    X, _, *_ = harmonize_local_files(data_dir)
    Y = regrid_glorys_target(data_dir)
    return X, Y
