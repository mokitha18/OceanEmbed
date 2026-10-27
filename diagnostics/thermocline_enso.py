"""
OceanEmbed - Diagnostics Module for Subsurface Thermocline Displacement & ENSO Tilt
MoES / INCOIS Review 2 Feature Addition

Calculates:
1. 20°C Isotherm (D20) depth field across the North Indian Ocean basin (0–1000m).
2. Basin zonal thermocline tilt between the Western Indian Ocean (50°E–70°E)
   and Eastern Bay of Bengal (85°E–100°E).
3. ENSO teleconnection classification (La Niña, El Niño, or ENSO-Neutral)
   and Subsurface Enthalpy Storage Status.
4. East-West Zonal Transect (12.0°N vertical cross-section, 0–300m, 50°E–95°E)
   with explicit 20°C isotherm tracking.
"""

import numpy as np
from scipy.interpolate import interp1d

def calculate_d20_grid(cube, depths, land_mask=None):
    """
    Computes the 20°C Isotherm (D20) depth in meters for every horizontal point (lat, lon).
    
    Args:
        cube: 3D numpy array of shape (15, n_lat, n_lon) representing temperature field.
        depths: 1D array/list of 15 standard depths: [0, 5, ..., 1000].
        land_mask: 2D boolean array (n_lat, n_lon), True for land cells.
        
    Returns:
        d20_grid: 2D numpy array (n_lat, n_lon) of D20 depth in meters (NaN for land).
    """
    if cube.ndim == 4:
        cube = cube[0]
        
    n_depth, n_lat, n_lon = cube.shape
    depths = np.asarray(depths, dtype=float)
    d20_grid = np.full((n_lat, n_lon), np.nan, dtype=float)
    
    for j in range(n_lat):
        for i in range(n_lon):
            if land_mask is not None and land_mask[j, i]:
                continue
                
            col_t = cube[:, j, i]
            if np.all(np.isnan(col_t)) or np.all(col_t == 0):
                continue
            
            # If surface is already cooler than 20°C
            if col_t[0] < 20.0:
                d20_grid[j, i] = 0.0
                continue
                
            # If even deepest depth is warmer than 20°C
            if col_t[-1] >= 20.0:
                d20_grid[j, i] = depths[-1]
                continue
            
            # Find vertical interval [z0, z1] crossing 20.0°C
            found = False
            for k in range(n_depth - 1):
                t0, t1 = col_t[k], col_t[k + 1]
                if t0 >= 20.0 and t1 <= 20.0:
                    z0, z1 = depths[k], depths[k + 1]
                    if abs(t1 - t0) > 1e-6:
                        d20_val = z0 + (20.0 - t0) * (z1 - z0) / (t1 - t0)
                    else:
                        d20_val = z0
                    d20_grid[j, i] = float(np.clip(d20_val, z0, z1))
                    found = True
                    break
                    
            if not found:
                # Fallback monotonic interpolation
                try:
                    f_interp = interp1d(col_t, depths, bounds_error=False, fill_value="extrapolate")
                    d20_grid[j, i] = float(np.clip(f_interp(20.0), 0.0, float(depths[-1])))
                except Exception:
                    d20_grid[j, i] = np.nan
                    
    return d20_grid

def evaluate_basin_enso_tilt(d20_grid, lats, lons, land_mask=None):
    """
    Evaluates the basin-wide zonal tilt between the Western Indian Ocean (50°E–70°E)
    and the Eastern Bay of Bengal (85°E–100°E).
    
    Teleconnection Thresholds:
    - La Niña: Eastern D20 > Western D20 + 15m (Deepened thermocline in Bay of Bengal)
    - El Niño: Western D20 > Eastern D20 + 15m (Shoaled thermocline in Bay of Bengal)
    - ENSO-Neutral: |Eastern D20 - Western D20| <= 15m (Equilibrium tilt)
    
    Returns:
        dict: Diagnostics dictionary containing metrics, state labels, and descriptive briefs.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    
    # Define bounding masks (focusing on the active tropical zonal band 5°N–15°N)
    lat_band = (lats >= 5.0) & (lats <= 18.0)
    
    lon_west = (lons >= 50.0) & (lons <= 70.0)
    lon_east = (lons >= 85.0) & (lons <= 100.0)
    
    mask_west = np.outer(lat_band, lon_west)
    mask_east = np.outer(lat_band, lon_east)
    
    if land_mask is not None:
        mask_west = mask_west & (~land_mask)
        mask_east = mask_east & (~land_mask)
        
    vals_west = d20_grid[mask_west]
    vals_east = d20_grid[mask_east]
    
    valid_west = vals_west[~np.isnan(vals_west)]
    valid_east = vals_east[~np.isnan(vals_east)]
    
    mean_west = float(np.mean(valid_west)) if len(valid_west) > 0 else 60.0
    mean_east = float(np.mean(valid_east)) if len(valid_east) > 0 else 75.0
    
    zonal_tilt = mean_east - mean_west
    
    # Evaluate ENSO state based on exact 15m directive
    if mean_east > mean_west + 15.0:
        enso_state = "La Niña"
        enthalpy_status = "High Storage - Cyclone Primer"
        status_color = "#0284c7" # Vibrant Ocean Cyan/Blue
        state_icon = "🌊"
        state_desc = (
            "Downwelling Kelvin waves have deepened the Bay of Bengal thermocline, "
            "insulating surface waters from sub-surface cooling and priming high upper-ocean enthalpy."
        )
    elif mean_west > mean_east + 15.0:
        enso_state = "El Niño"
        enthalpy_status = "Upwelling Suppressed"
        status_color = "#dc2626" # Warning Crimson
        state_icon = "⚠️"
        state_desc = (
            "Western basin thermocline is significantly deepened while the Bay of Bengal thermocline "
            "has shoaled, suppressing normal equatorial upwelling mechanisms."
        )
    else:
        enso_state = "ENSO-Neutral"
        enthalpy_status = "Balanced Equilibrium"
        status_color = "#16a34a" # Stable Emerald Green
        state_icon = "⚖️"
        state_desc = (
            "Zonal thermocline tilt is within normal equilibrium bounds (|ΔD20| < 15m), "
            "supporting balanced monsoon and oceanic circulation."
        )
        
    # Bay of Bengal D20 qualification (Normal ~55m, Deepened >75m)
    if mean_east > 75.0:
        bob_depth_label = f"{mean_east:.1f} m (Deepened > 75m)"
    elif mean_east < 60.0:
        bob_depth_label = f"{mean_east:.1f} m (Normal ~55m)"
    else:
        bob_depth_label = f"{mean_east:.1f} m (Moderate 60–75m)"
        
    return {
        "enso_state": enso_state,
        "mean_west_d20": mean_west,
        "mean_east_d20": mean_east,
        "zonal_tilt": zonal_tilt,
        "enthalpy_status": enthalpy_status,
        "bob_depth_label": bob_depth_label,
        "status_color": status_color,
        "state_icon": state_icon,
        "state_desc": state_desc
    }

def extract_zonal_transect(cube, depths, lats, lons, target_lat=12.0, min_lon=50.0, max_lon=95.0, max_depth=300.0, land_mask=None):
    """
    Extracts a 2D vertical cross-section along Latitude 12.0°N
    (Longitudes 50°E to 95°E, Depths 0–300m) for subsurface visual exploration.
    
    Returns:
        transect_temps: 2D array of shape (n_sub_depths, n_sub_lons)
        sub_depths: 1D array of depth levels <= max_depth
        sub_lons: 1D array of longitudes between min_lon and max_lon
        d20_transect: 1D array of D20 isotherm depth along this transect
        actual_lat: float, nearest latitude matched
    """
    if cube.ndim == 4:
        cube = cube[0]
        
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    depths = np.asarray(depths, dtype=float)
    
    # Find nearest latitude index to 12.0°N
    lat_idx = int(np.abs(lats - target_lat).argmin())
    actual_lat = float(lats[lat_idx])
    
    # Filter depth levels up to max_depth (e.g. 300m)
    depth_mask = depths <= max_depth
    sub_depths = depths[depth_mask]
    n_sub_depths = len(sub_depths)
    
    # Filter longitudes (50°E to 95°E)
    lon_mask = (lons >= min_lon) & (lons <= max_lon)
    sub_lons = lons[lon_mask]
    n_sub_lons = len(sub_lons)
    
    # Extract vertical slice: (n_sub_depths, n_sub_lons)
    transect_temps = cube[depth_mask, lat_idx, :][:, lon_mask].copy()
    
    # Mask land cells if provided
    if land_mask is not None:
        slice_land = land_mask[lat_idx, lon_mask]
        for i_l, is_land in enumerate(slice_land):
            if is_land:
                transect_temps[:, i_l] = np.nan
                
    # Calculate D20 line along this transect
    d20_transect = np.full(n_sub_lons, np.nan, dtype=float)
    for i_l in range(n_sub_lons):
        col_t = transect_temps[:, i_l]
        if np.all(np.isnan(col_t)) or np.all(col_t == 0):
            continue
        if col_t[0] < 20.0:
            d20_transect[i_l] = 0.0
            continue
        if col_t[-1] >= 20.0:
            d20_transect[i_l] = sub_depths[-1]
            continue
            
        for k in range(n_sub_depths - 1):
            t0, t1 = col_t[k], col_t[k + 1]
            if t0 >= 20.0 and t1 <= 20.0:
                z0, z1 = sub_depths[k], sub_depths[k + 1]
                if abs(t1 - t0) > 1e-6:
                    d20_transect[i_l] = z0 + (20.0 - t0) * (z1 - z0) / (t1 - t0)
                else:
                    d20_transect[i_l] = z0
                break
                
    return transect_temps, sub_depths, sub_lons, d20_transect, actual_lat
