import numpy as np
from .. import models

def test_app_mag_model():
    """
    Testing lambda handler
    """
    df={}
    df['phot_rp_mean_mag'] = np.array([8.0,9.0,10.0,12.0,14.0])
    df['phot_bp_mean_mag'] = df['phot_rp_mean_mag']+1.0
    df['feh'] = np.array([-0.5, 0.25, 0.0, 0.25, 0.5])
    df['ag'] = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    df['parallax'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    x1=np.array([8.24388813,  9.25270414, 10.226809,   12.21172063, 14.18093982])
    x2=np.array([9.44964953, 10.4497579,  11.45449065, 13.47381074, 15.47384311])
    x3=np.array([10361441.69586165,  4124552.54325019,  1634871.94275711,   254539.80432366, 40340.63724908])
    x4=1.0073027045309626
    # rmodel1=roman_magnitudes.models.RomanAppMagModel().load(df)
    rmodel1=models.RomanAppMagModel().load(df)
    assert np.allclose(x1,rmodel1.get_magnitude('roman_f106'))    

    df={}
    df['tmass_ks'] = np.array([8.0,9.0,10.0,12.0,14.0])
    df['tmass_j'] = df['tmass_ks']+0.5
    df['parallax'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    rmodel1.load(df)    
    assert np.allclose(x2,rmodel1.get_magnitude('roman_grism'))    
    assert np.allclose(x3, rmodel1.mag_to_counts(rmodel1.get_magnitude('roman_grism'),'roman_grism'))
    assert x4 == rmodel1.mag_to_counts(26.61,'roman_f062')


def test_abs_mag_model():
    """
    Testing lambda handler
    """
    rmodel2=models.RomanAbsMagModel()
    feh=np.array([-0.5, 0.25, 0.0, 0.25])
    teff=np.array([5000, 4500, 4250.0, 4000])
    logg=np.array([4.0, 3.5, 3.0, 0.0])
    x1=np.array([-0.094432, -0.094432, -0.094432, -0.094432])
    x2=np.array([0.3718707,  0.54921722, 0.62130004, 0.807523])
    x3=np.array([-2.10138462, -2.58735215, -2.87810552, -3.18653255])
    x4=np.array([-0.12212598, -0.12352369, -0.10399857, -0.09201436])
    assert np.allclose(x1,rmodel2.get_magnitude(feh, teff, logg, bandpass='gaia_g'))
    assert np.allclose(x2,rmodel2.get_magnitude(feh, teff, logg, bandpass='gaia_gbp'))
    assert np.allclose(x3,rmodel2.get_magnitude(feh, teff, logg, bandpass='tmass_ks'))
    assert np.allclose(x4,rmodel2.get_magnitude(feh, teff, logg, bandpass='roman_f062'))

