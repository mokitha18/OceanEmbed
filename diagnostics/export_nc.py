import xarray as xr

def export_to_netcdf(cube, depths, lats, lons, output_filename="oceanembed_output.nc", time_val=None):
    """
    Converts model output into CF-compliant NetCDF-4 (.nc) files.
    """
    if time_val is None:
        time_val = [0]
        
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["time", "depth", "lat", "lon"], cube[None, ...])
        ),
        coords=dict(
            time=time_val,
            depth=(["depth"], depths),
            lat=(["lat"], lats),
            lon=(["lon"], lons),
        ),
        attrs=dict(
            description="OceanEmbed Reconstructed Subsurface Temperature",
            units="degC",
            creator="OceanEmbed Pipeline"
        ),
    )
    
    ds.to_netcdf(output_filename)
    print(f"Exported to {output_filename}")
