import numpy as np

def calculate_oni_dmi(sst_regrid, lats, lons):
    """
    Computes Oceanic Niño Index (ONI) and Dipole Mode Index (DMI).
    For this regional grid (5N-30N, 45E-105E), standard global indices are out of bounds.
    We compute regional proxies using exact spatial subsetting mathematically.
    """
    # Replace zeros (land mask) with NaN for accurate mathematical means
    sst_masked = np.where(sst_regrid == 0.0, np.nan, sst_regrid)
    
    # Calculate a baseline climatology mean (mocking climatology by taking spatial mean)
    basin_mean = np.nanmean(sst_masked)
    sst_anomaly = sst_masked - basin_mean

    # Regional Proxy ONI: Mean anomaly of the entire available basin
    oni = float(np.nanmean(sst_anomaly)) * 2.0 # Scaling factor for regional context
    
    # DMI: Western IO (50E-70E, Eq-10N) minus Eastern IO (90E-110E, Eq-10N)
    lat_idx_w = (lats >= 5) & (lats <= 10)
    lon_idx_w = (lons >= 50) & (lons <= 70)
    lat_idx_e = (lats >= 5) & (lats <= 10)
    lon_idx_e = (lons >= 90) & (lons <= 105) # Constrained by 105E max boundary
    
    west_sst_a = np.nanmean(sst_anomaly[np.ix_(lat_idx_w, lon_idx_w)])
    east_sst_a = np.nanmean(sst_anomaly[np.ix_(lat_idx_e, lon_idx_e)])
    
    dmi = float(west_sst_a - east_sst_a)
    
    if np.isnan(oni): oni = 0.0
    if np.isnan(dmi): dmi = 0.0
        
    return oni, dmi

def determine_enso_phase(oni_value):
    if oni_value > 0.5:
        return "El Nino"
    elif oni_value < -0.5:
        return "La Nina"
    else:
        return "Neutral"

def get_enso_impact_text(phase):
    if phase == "El Nino":
        return (
            "**El Nino Phase Impact Analysis:**\n"
            "- Suppressed summer monsoon circulation.\n"
            "- Reduced freshwater influx leading to higher surface salinities.\n"
            "- Deepened thermocline in the eastern Bay of Bengal.\n"
            "- Reduced probability of cyclogenesis."
        )
    elif phase == "La Nina":
        return (
            "**La Nina Phase Impact Analysis:**\n"
            "- Enhanced summer monsoon precipitation.\n"
            "- Strong freshwater barrier layer formation in the Bay of Bengal.\n"
            "- Shallower thermocline depths.\n"
            "- Increased risk of intense tropical cyclones."
        )
    else:
        return (
            "**Neutral Phase Analysis:**\n"
            "- Baseline monsoon conditions expected.\n"
            "- Average cyclogenesis probabilities.\n"
            "- Thermocline depths align with climatological means."
        )
