# Computing Roman magnitudes in different photometric bands
Two models are provided one computes the apparent magnitude while the other computes Absolute magnitude. 
Output is available for following roman passbands 

*['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129', 'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213', 'roman_prism', 'roman_grism', 'roman_prism_b8', 'roman_prism_r8', 'roman_grism_b8', 'roman_grism_r8']*

### Apparent Magnitudes (class RomanAppMagModel)
Apparent magnitude is based on analytical formulas and requires photometry in Gaia or 2MASS bands/
One can also provide additional stellar parameters like metallicity and extinction. 
This leads to 4 basic combinations or schemes. Optionally, for each of the scheme given below one can also provide parallax.

- scheme A: *('phot_bp_mean_mag', 'phot_rp_mean_mag', 'feh', 'ag')*
- scheme B: *('phot_bp_mean_mag', 'phot_rp_mean_mag')*
- scheme C: *('tmass_j', 'tmass_ks')*
- scheme D: *('phot_g_mean_mag')*

 
### Absolute Magnitudes (class RomanAbsMagModel)
Absolute magntidues are presented as a function of stellar parameters *(feh, teff, logg)*. They are in AB magnitude system for Roman bands and Vega for 
other bands *(gaia_gbp, gaia_grp, gaia_g, tmass_j, tmass_h, tmass_ks, wise_w1, wise_w2)*. 
Results are based on interpolation tables constructed from Phoenix synthetic spectra using the python synphot package.
The magnitudes are **normalized to have a magnitude of 0.0 in *gaia_g* band**.


### Usagae
```python
import numpy as np
import roman_magnitudes.models
df={}
n=10
df['phot_bp_mean_mag']=np.random.uniform(size=n)+14
df['phot_rp_mean_mag']=np.random.uniform(size=n)+14
df['feh']=np.random.uniform(size=n)-0.5
df['ag']=np.random.uniform(size=n)
df['parallax']=1+np.random.uniform(size=n)
rmodel1=roman_magnitudes.models.RomanAppMagModel().load(df)
print('roman_f106:',rmodel1.get_magnitude('roman_f106'))

df={}
n=10
df['tmass_j']=np.random.uniform(size=n)+14
df['tmass_ks']=np.random.uniform(size=n)+14
df['parallax']=1+np.random.uniform(size=n)
rmodel1.load(df)
print('roman_grism:',rmodel1.get_magnitude('roman_grism'))
print('roman_grism counts:',rmodel1.mag_to_counts(rmodel1.get_magnitude('roman_grism'),'roman_grism'))
print('roman_grism counts:',rmodel1.mag_to_counts(26.61,'roman_f062'))


rmodel2=roman_magnitudes.models.RomanAbsMagModel()
feh=np.array([-0.5, 0.25, 0.0, 0.25])
teff=np.array([5000, 4500, 4250.0, 4000])
logg=np.array([4.0, 3.5, 3.0, 0.0])
print(rmodel2.get_magnitude(feh, teff, logg, bandpass='gaia_g'))
print(rmodel2.get_magnitude(feh, teff, logg, bandpass='gaia_gbp'))
print(rmodel2.get_magnitude(feh, teff, logg, bandpass='tmass_ks'))
print(rmodel2.get_magnitude(feh, teff, logg, bandpass='roman_f062'))

