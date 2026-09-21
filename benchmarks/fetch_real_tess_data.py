"""
Download real TESS light curves for confirmed exoplanet hosts.
Uses astroquery to fetch data from MAST.

Target stars: hosts of WASP-17b, TOI-132b, HD 209458b, L 98-59b, TRAPPIST-1b
"""

import sys
sys.path.insert(0, 'src')

from astroquery.mast import Observations
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Confirmed exoplanet hosts with known transit parameters
TARGETS = {
    'WASP-17': {
        'tic_id': 36734222,
        'planet': 'WASP-17b',
        'period_days': 3.7354,
        'duration_hours': 4.35,
        'depth': 0.01056,
    },
    'TOI-132': {
        'tic_id': 59873312,
        'planet': 'TOI-132b',
        'period_days': 1.9621,
        'duration_hours': 2.05,
        'depth': 0.00087,
    },
    'HD 209458': {
        'tic_id': 420814525,
        'planet': 'HD 209458b',
        'period_days': 3.5235,
        'duration_hours': 3.0,
        'depth': 0.01474,
    },
    'L 98-59': {
        'tic_id': 307210830,
        'planet': 'L 98-59b',
        'period_days': 3.4043,
        'duration_hours': 1.6,
        'depth': 0.00049,
    },
    'TRAPPIST-1': {
        'tic_id': 278956474,
        'planet': 'TRAPPIST-1b',
        'period_days': 1.5138,
        'duration_hours': 1.1,
        'depth': 0.00726,
    },
}

def download_tess_light_curve(tic_id: int, output_dir: Path = Path('data/tess_lcurves')):
    """
    Download 2-minute cadence TESS light curve from MAST.
    
    Returns:
        DataFrame with columns: time, flux, flux_err
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Query MAST for TESS observations of this TIC ID
        obs = Observations.query_criteria(
            obs_collection='TESS',
            target_name=str(tic_id),
            dataproduct_type='timeseries',
        )
        
        if len(obs) == 0:
            raise ValueError(f"No TESS observations found for TIC {tic_id}")
        
        # Get product list
        obsids = obs['obsid']
        products = Observations.get_product_list(obsids)
        
        # Filter for light curve FITS files
        fits_files = []
        for i in range(len(products)):
            fname = str(products['productFilename'][i])
            if fname.endswith('lc.fits'):
                fits_files.append(products[i])
        
        if len(fits_files) == 0:
            raise ValueError(f"No light curve files found for TIC {tic_id}")
        
        # Download first file
        first_file = fits_files[0]
        result = Observations.download_products(
            [str(first_file['obsID'])],
            mrp_only=False,
            cache=True,
        )
        
        if len(result) == 0:
            raise ValueError(f"Download failed for TIC {tic_id}")
        
        # Parse FITS
        from astropy.io import fits
        local_path = str(result['Local Path'][0])
        
        with fits.open(local_path) as hdul:
            data = hdul[1].data
            
            # Extract time and flux
            time = data['TIME']
            flux = data['SAP_FLUX']
            flux_err = data['SAP_FLUX_ERR']
            
            # Normalize flux
            flux_median = np.nanmedian(flux)
            flux = flux / flux_median
            flux_err = flux_err / flux_median
            
            df = pd.DataFrame({
                'time': time,
                'flux': flux,
                'flux_err': flux_err,
            })
            
            # Remove NaNs
            df = df.dropna()
            
            if len(df) == 0:
                raise ValueError(f"No valid data points after removing NaNs for TIC {tic_id}")
            
            # Save
            output_path = output_dir / f"{tic_id}_lcurve.csv"
            df.to_csv(output_path, index=False)
            
            return df
    
    except Exception as e:
        raise Exception(f"Failed to download TIC {tic_id}: {str(e)}")

def main():
    print("Downloading TESS light curves for confirmed exoplanet hosts...")
    print()
    
    for name, params in TARGETS.items():
        print(f"  {name} (TIC {params['tic_id']})...", end=' ', flush=True)
        try:
            lcurve = download_tess_light_curve(params['tic_id'])
            print(f"✓ {len(lcurve)} points")
        except Exception as e:
            print(f"✗ {e}")

if __name__ == '__main__':
    main()