import numpy as np
from scipy.interpolate import RegularGridInterpolator
import ebf
from importlib.resources import files

class RomanAppMagModel:
    """
    Model to compute apparent magnitude in Roman pass bands (filter) given photometry and stellar params like 
        feh: Metallicity [M/H], mh_xgboost based on Gaia XP spectra
        ag: Extinction, ag_gspphot from Gaia
        parallax: in milli-arcsec from Gaia
    
    Output is available for following roman passbands 
    ['roman_f062',     'roman_f087',     'roman_f106',     'roman_f129',  'roman_f146', 
     'roman_f158',     'roman_f184',     'roman_f213',     'roman_prism', 'roman_grism', 
     'roman_prism_b8', 'roman_prism_r8', 'roman_grism_b8', 'roman_grism_r8']

    
    Input data can is specified  using the method load() via a dict representing a dataframe.
    Allowed key combinations in data frame are given below, for each of the schemes optionally 
    one can also include 'parallax'.
    ---------------------------
        scheme A: ['phot_bp_mean_mag', 'phot_rp_mean_mag', 'feh', 'ag']
        scheme B: ['phot_bp_mean_mag', 'phot_rp_mean_mag']
        scheme C: ['tmass_j', 'tmass_ks']
        scheme D: ['phot_g_mean_mag']
    
    Example
    ---------
    df={}
    df['phot_rp_mean_mag'] = np.array([8.0,9.0,10.0,12.0,14.0])
    df['phot_bp_mean_mag'] = df['phot_rp_mean_mag']+1.0
    df['feh'] = np.array([-0.5, 0.25, 0.0, 0.25, 0.5])
    df['ag'] = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    df['parallax'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])    
    rmodel1=RomanAppMagModel().load(df)
    print('roman_f106:',rmodel1.get_magnitude('roman_f106'))
    
    df={}
    df['tmass_ks'] = np.array([8.0,9.0,10.0,12.0,14.0])
    df['tmass_j'] = df['tmass_ks']+0.5
    df['parallax'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    rmodel1.load(df)
    print('roman_grism:',rmodel1.get_magnitude('roman_grism'))
    print('roman_grism counts:',rmodel1.mag_to_counts(rmodel1.get_magnitude('roman_grism'),'roman_grism'))
    print('roman_grism counts:',rmodel1.mag_to_counts(26.61,'roman_f062'))
    

    Author: Sanjib Sharma, Dec 2025
    """
    
    def __init__(self):
        datadir = files('roman_magnitudes.data')
        self.data = np.genfromtxt(datadir.joinpath('all_schemes_coefficients.csv'), delimiter=',', dtype=None, names=True, encoding='utf-8',autostrip=True)
        data = np.genfromtxt(datadir.joinpath('zero_points_20250707.csv'), delimiter=',', dtype=None, names=True, encoding='utf-8',autostrip=True)
        self.zero_point={'roman_'+data['band'][i].lower(): data['Z_R'][i] for i in range(data['Z_R'].size)}

        self.coeff={}
        for j in range(self.data['band'].size):
            temp=f"{self.data['scheme'][j]}_roman_{self.data['band'][j]}_{self.data['l'][j]}"
            self.coeff[temp]=np.array([self.data[f"c{i}"][j] for i in range(9)])
        
        
    def load(self, df):
        """
        Input new data frame
        """
        set_A=set(['phot_bp_mean_mag', 'phot_rp_mean_mag', 'feh', 'ag'])
        set_B=set(['phot_bp_mean_mag', 'phot_rp_mean_mag'])
        set_C=set(['tmass_j', 'tmass_ks'])
        set_D=set(['phot_g_mean_mag'])
        if type(df) is dict:
            set_df=set(df.keys())
        else:
            set_df=set(df.dtype.names)
        self.fg=None
        
        # if set_A1.issubset(set_df):
        #     self.scheme='A1'
        #     self.app_mag=d['phot_rp_mean_mag']
            
        #     col=d['phot_bp_mean_mag']-d['phot_rp_mean_mag']            
            
        
        if set_A.issubset(set_df):
            self.scheme='A'
            self.mag1=df['phot_bp_mean_mag']
            self.mag2=df['phot_rp_mean_mag']
            psystem='gaia'
        elif set_B.issubset(set_df):
            self.scheme='B'
            self.mag1=df['phot_bp_mean_mag']
            self.mag2=df['phot_rp_mean_mag']
            psystem='gaia'
        elif set_C.issubset(set_df):
            self.scheme='C'
            self.mag1=df['tmass_j']
            self.mag2=df['tmass_ks']
            psystem='tmass'
        elif set_D.issubset(set_df):
            self.scheme='D'
            self.mag1=df['phot_g_mean_mag']
            self.mag2=df['phot_g_mean_mag']
            psystem=''
        col=self.mag1-self.mag2            
            
        if 'feh' in set_df:
            feh=df['feh']
        else:
            feh=np.zeros_like(col)
        
        if 'ag' in set_df:
            ag=df['ag']
        else:
            ag=np.zeros_like(col)
            
        self.fg=np.zeros_like(col)+np.nan
        if 'parallax' in set_df:        
            cond=np.isfinite(df['parallax'])&(df['parallax']>0)
            self.fg[cond]=self._giant_prob(self.mag1[cond], self.mag2[cond],self._omega2dmod(df['parallax'][cond]*1e-3),psystem=psystem)  

        self.X=np.vstack([np.ones(len(col))*1.0, col, col**2, col**3, col**4, feh, ag, ag*ag, ag*col]).T

        return self

    def _omega2dmod(self, omega):
        """
        Arguments:
            omega (float)- parallax in arcsec
        Returns:
            dmod (float)- distance modulus     
        """
        return 5.0*np.log10(100.0*1e-3/omega)
        
    def get_magnitude(self, roman_bandpass):        
        roman_banpass=roman_bandpass.lower()
        coeff_n=self.coeff[f"{self.scheme}_{roman_bandpass}_{'none'}"]
        temp=(self.X @ coeff_n)        
        
        cond=np.isfinite(self.fg)
        if np.sum(cond) > 0:
            X=self.X[cond,:]
            fg=self.fg[cond]
            coeff_g=self.coeff[f"{self.scheme}_{roman_bandpass}_{'giant'}"]
            coeff_d=self.coeff[f"{self.scheme}_{roman_bandpass}_{'dwarf'}"]
            temp[cond]=(X @ coeff_g)*fg+(X @ coeff_d)*(1-fg)            
            
        return self.mag2+temp            
                               
    def _giant_absmag(self,col,psystem='tmass'):
        """
        Arguments
        ---------
        col: (j-ks) or (bp-rp)
        
        Output
        ------
        gaia_rp or tmass_ks
        """
        # aebv=gutil.aebv_factor()        
        if psystem == 'tmass':
            # m=aebv['tmass_ks']/(aebv['tmass_j']-aebv['tmass_ks'])        
            m=0.6926277 
        elif psystem == 'gaia':
            # m=aebv['gaia_grp']/(aebv['gaia_gbp']-aebv['gaia_grp'])        
            m=1.4165213
        else:
            raise RuntimeError("psystem should be 'tmass' or 'gaia' ")
        return m*col
    
    def _giant_prob(self, m1, m2, dmod, psystem='tmass'):
        """
        Probability to be a giant 
        Based on a claissification line in (abs_mag, color) space        
        """
        if psystem in ['tmass', 'gaia']:
            absmag=self._giant_absmag(m1-m2,psystem=psystem)
            return 0.5*(np.tanh((absmag-(m2-dmod))/0.5)+1)
        else:
            return np.zeros_like(m1)+np.nan

    def mag_to_counts(self, roman_magnitude, roman_bandpass):
        """
        Count rate in e/s for a given roman bandpass
        """
        temp=(self.zero_point[roman_bandpass]-roman_magnitude)/2.5
        return np.power(10.0,temp)

class RomanAbsMagModel:
    """
    Absolute magntidue in various bandpasses as a function of (feh, teff, logg). 
    Magnitudes are in AB magnitude system for Roman and Vega for others. 
    Results are based on Phoenix synthetic spectra wrapped in python synphot package.
    The magnitudes are normalized to have a magnitude of 0.0 in 'gaia_g' band.
    Magnitudes are available for following bands.
    
    ['gaia_g', 'tmass_j', 'tmass_h', 'tmass_ks', 'wise_w1', 'wise_w2', 'gaia_bp', 
        'gaia_rp', 'roman_f062', 'roman_f087', 'roman_f106', 'roman_f129', 'roman_f146', 
        'roman_f158', 'roman_f184', 'roman_f213', 'roman_prism', 'roman_grism', 'roman_prism_b8', 
        'roman_prism_r8', 'roman_grism_b8', 'roman_grism_r8']
            
    
    Example
    --------
    rmodel2=RomanAbsMagModel()
    feh=np.array([-0.5, 0.25, 0.0, 0.25])
    teff=np.array([5000, 4500, 4250.0, 4000])
    logg=np.array([4.0, 3.5, 3.0, 0.0])
    ebv=
    print(rmodel2(feh, teff, logg, bandpass='gaia_g'))
    print(rmodel2(feh, teff, logg, bandpass='gaia_bp'))
    print(rmodel2(feh, teff, logg, bandpass='roman_f062'))

    Author: Sanjib Sharma, Dec 2025
    """
    def __init__(self):        
        datadir = files('roman_magnitudes.data')
        model_photo_grid = ebf.read(datadir.joinpath('model_photometry_full_grid_mh_shift_new_edge.ebf'),'/')
        all_bps = [key for key in model_photo_grid.keys() if '_abmags' in key]
        
        inds = np.lexsort((model_photo_grid['loggs'], model_photo_grid['teffs'], model_photo_grid['mhs']))
        
        umh = np.unique(model_photo_grid['mhs'][inds])
        uT = np.unique(model_photo_grid['teffs'][inds])
        ulogg = np.unique(model_photo_grid['loggs'][inds])
        
        # Tg, mhg, lgg = np.meshgrid(umh, uT, ulogg, indexing='ij')    
        interpolated_photometry = {}
        shape=[len(umh), len(uT), len(ulogg)]
        for bp_key in all_bps:
            data = model_photo_grid[bp_key][inds]
        #        if bp_key != 'g_abmags':
            data = data - model_photo_grid['g_abmags'][inds]
            data = data.reshape(shape)
            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    mask = np.isnan(data[i,j,:])
                    if np.any(mask):
                        arr = np.copy(data[i,j,:])
                        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
                        data[i,j,:] = arr
            
            interpolated_photometry[bp_key] = RegularGridInterpolator((umh, uT, ulogg), data, bounds_error=False)
        inkeys=['g_abmags', 'j_abmags', 'h_abmags', 'ks_abmags', 'w1_abmags', 'w2_abmags', 'bp_abmags', 
                'rp_abmags', 'f062_abmags', 'f087_abmags', 'f106_abmags', 'f129_abmags', 'f146_abmags', 
                'f158_abmags', 'f184_abmags', 'f213_abmags', 'prism_abmags', 'grism_abmags', 'prism_b8_abmags', 
                'prism_r8_abmags', 'grism_b8_abmags', 'grism_r8_abmags']
        outkeys=['gaia_g', 'tmass_j', 'tmass_h', 'tmass_ks', 'wise_w1', 'wise_w2', 'gaia_gbp', 
                'gaia_grp', 'roman_f062', 'roman_f087', 'roman_f106', 'roman_f129', 'roman_f146', 
                'roman_f158', 'roman_f184', 'roman_f213', 'roman_prism', 'roman_grism', 'roman_prism_b8', 
                'roman_prism_r8', 'roman_grism_b8', 'roman_grism_r8']
        self.rename_cols(interpolated_photometry, inkeys, outkeys)
        
        self.interpolated_photometry=interpolated_photometry
        
        ab2vega={}
        ab2vega['tmass_j']  = 0.91338 +0.046
        ab2vega['tmass_h']  = 1.39118 +0.000 
        ab2vega['tmass_ks'] = 1.86344 +0.032
        ab2vega['gaia_g']   = 0.127432-0.033 
        ab2vega['gaia_gbp'] = 0.02385 -0.047
        ab2vega['gaia_grp'] = 0.38154 -0.022 
        ab2vega['wise_w1']  = 2.69427 +0.054 
        ab2vega['wise_w2']  = 3.33378 +0.034 
        # ab2vega['wise_w3']=5.16903 
        # ab2vega['wise_w4']=6.64459
        self.ab2vega=ab2vega
        
        data = np.genfromtxt(datadir.joinpath('aebv_factor.csv'), delimiter=',', dtype=None, names=True, encoding='utf-8',autostrip=True)
        self.aebv={data['band'][i].lower(): data['aebv'][i] for i in range(data['band'].size)}
        
        
    def rename_cols(self, data, inkeys, outkeys, delete=True):    
        for i in range(len(inkeys)):
            data[outkeys[i]]=data[inkeys[i]]
            if delete:
                del data[inkeys[i]]
                
    def get_magnitude(self, feh, teff, logg, ebv=0.0, bandpass='roman_f062'):
        """
        Arguments
        ----------
        feh: array(float)
        teff: array(float)
        logg: array(float)
        
        Keywords
        ----------
        band: str 
        ebv: array(float), extinction default is 0.0
        """
        
        input_tuple = (feh, teff, logg)
        temp=self.interpolated_photometry[bandpass](input_tuple)+self.aebv[bandpass]*ebv
        if bandpass in self.ab2vega:
            temp=temp-self.ab2vega[bandpass]
            
        return temp
