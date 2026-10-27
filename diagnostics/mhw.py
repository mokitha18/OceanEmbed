import numpy as np
import pandas as pd

def detect_marine_heatwaves(cube_50m_timeseries):
    """
    Implements Hobday et al. (2016) Marine Heatwave detection logic.
    Flags areas where reconstructed subsurface temperature at 50m depth
    exceeds the 90th percentile of historical climatology for >= 5 consecutive days.
    
    Args:
        cube_50m_timeseries (numpy.ndarray): Reconstructed temperature at 50m shape (time, lat, lon).
        
    Returns:
        numpy.ndarray: Boolean mask of active MHW events at the last time step.
    """
    if len(cube_50m_timeseries) < 5:
        # Not enough temporal data to calculate a 5 day window
        t_len, n_lat, n_lon = cube_50m_timeseries.shape
        return np.zeros((n_lat, n_lon), dtype=bool)

    t_len, n_lat, n_lon = cube_50m_timeseries.shape
    
    # Calculate the 90th percentile threshold across the time dimension
    threshold_90th = np.nanpercentile(cube_50m_timeseries, 90, axis=0)
    
    # Identify exceedances
    is_exceeding = cube_50m_timeseries > threshold_90th
    
    mhw_active = np.zeros((n_lat, n_lon), dtype=bool)
    
    for j in range(n_lat):
        for i in range(n_lon):
            series = is_exceeding[:, j, i]
            rolling_sum = pd.Series(series).rolling(window=5, min_periods=5).sum()
            # If the most recent rolling sum is 5, then it's an active MHW
            if rolling_sum.iloc[-1] == 5.0:
                mhw_active[j, i] = True
                
    return mhw_active
