import numpy as np
import scipy.integrate as integrate

def calculate_tchp(temperature_profile, depths, rho=1025, cp=3985):
    """
    Computes Tropical Cyclone Heat Potential (TCHP).
    TCHP = rho * Cp * int_{0}^{D26} (T(z) - 26) dz
    """
    if np.all(np.isnan(temperature_profile)) or temperature_profile[0] < 26:
        return 0.0

    valid_indices = temperature_profile >= 26
    if not np.any(valid_indices):
        return 0.0
        
    # Find the depth of the 26C isotherm
    idx = np.where(valid_indices == False)[0]
    if len(idx) == 0:
        d26 = depths[-1]
        valid_depths = depths
        valid_temps = temperature_profile
    else:
        # Interpolate the exact depth of 26C
        idx = idx[0]
        if idx == 0:
            return 0.0
        z1, z2 = depths[idx-1], depths[idx]
        t1, t2 = temperature_profile[idx-1], temperature_profile[idx]
        if t1 == t2:
            z26 = z1
        else:
            z26 = z1 + (26 - t1) * (z2 - z1) / (t2 - t1)
            
        valid_depths = np.append(depths[:idx], z26)
        valid_temps = np.append(temperature_profile[:idx], 26.0)

    # Calculate integrand (T(z) - 26)
    integrand = valid_temps - 26.0
    
    # Integrate using Simpson's rule or trapezoidal rule
    integral_val = np.trapz(integrand, valid_depths)
    
    # Convert J/m^2 to kJ/cm^2 (divide by 1000 for kJ, divide by 10000 for cm^2)
    tchp = (rho * cp * integral_val) / 1e7
    return max(0.0, tchp)

def get_tchp_map(cube, depths):
    """
    Computes TCHP for an entire 3D cube (depth, lat, lon).
    Returns a 2D map of TCHP (lat, lon).
    """
    depths_m = np.array(depths)
    
    if cube.ndim == 4:
        cube = cube[0]
        
    n_depth, n_lat, n_lon = cube.shape
    tchp_map = np.zeros((n_lat, n_lon))
    
    for j in range(n_lat):
        for i in range(n_lon):
            prof = cube[:, j, i]
            if np.all(prof == 0.0) or np.all(np.isnan(prof)):
                tchp_map[j, i] = np.nan
            else:
                tchp_map[j, i] = calculate_tchp(prof, depths_m)
            
    return tchp_map
