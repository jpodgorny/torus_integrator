


# Torus reflection integrator

Create simple spectro-polarimetric table models with the help of this Python class.

## Purpose

This python module takes local spectro-polarimetric reflection tables from FITS files (described in Podgorný et al. 2022, available at https://doi.org/10.6084/m9.figshare.16726207). It puts a source of X-ray power-law in the center with energy binning, a power-law index Γ, and primary polarization according to the local reflection tables. The user defines a range of inner radii, torus half-opening angles and observer's inclinations, for which the final ASCII spectro-polarimetric tables are to be computed. Then the routine takes into account the visible and illuminated part of the inner walls of a pure circular torus and computes the integrated view for a distant observer, seeing a single reflection. Direct radiation or multiple scattering are not taken into account. The output tables show renormalized Stokes parameters I, Q, U dependent on energy, named according to the remaining model parameters and their values.

## Limitations

* only reflection from a pure circular torus can be computed at the moment
* the routines need to be adapted to take into account different reflection tables

## Dependencies

The module uses [AstroPy](https://www.astropy.org/) for reading the FITS files and storing the ASCII files.


## Usage

The basic usage involves importing the helper class from the module, creating an instace of `TorusModel` and filling the model with reflection tables. The skeleton looks like this:

```python
from torus_integrator import TorusModel

def get_energies_and_parameters(tables_directory, Gamma, e_min, e_max):
	return [] # low and high energy bins, indices for spectral extraction, parameter values from given reflection tables
#end def

def load_tables(tables_directory, name_IQU, first_ind, last_ind, collect):
	return [] # loaded spectra from given reflection tables
#end def

saving_directory = './model_tables'
tables_directory = '.'

# linear binning, not dynamic
N_u = ... # across (90,270) degrees, the other half is symmetrically added
N_v = ... # between the shadow line and the equatorial plane,
         # i.e. 180°-Theta <= v <= 180°

# primary polarization states "(name, pol. frac., pol. ang.)" to be computed
PPs = [...]

# mu_e emission inclination cosines from the pole to be computed
ms = [...]

# r_in parameters, the torus inner radii (arbitrary units), to be computed
inner_radii = [...]

# Theta half-opening angles to be computed from the pole in degrees
opening_angles = [...]

# which Stokes parameters to compute - if polarization, then specify 
# both 'Q' and 'U', even though Us will be zero in the end due to symmetry
names_IQUs = [...]

# Gammas primary power-law indices to be computed (need to be exactly as in 
# the loaded local tables)
Gammas = [...]

for Gamma in Gammas:
    E_min = ... # for low bin edge
    E_max = ... # for low bin edge
    # use any local reflection table to get the energy values and parameter 
    # values
    energies, first_ind, last_ind, parameters, collect \
            = get_energies_and_parameters(tables_directory, float(Gamma), \
                                                  E_min, E_max)
    # get the relevant spectra per Gamma
    all_spectra = []
    for name_iqu in names_IQUs:
        one_iqu = load_tables(tables_directory, name_iqu, first_ind, \
                                      last_ind, collect)
        all_spectra.append(one_iqu)
        
    # produce the table models
    for r_in_input in inner_radii:
        for Theta_input in opening_angles:
            # my first model
            TM = TorusModel(saving_directory, energies, parameters, \
                                all_spectra, Theta_input, r_in_input, N_u, \
                                N_v, names_IQUs, PPs, ms, Gamma)
            for g in TM.generator():
                name, ener_lo, ener_hi, final_spectra = g
                # save to the desired directory
                TM.save_ascii(name, ener_lo, ener_hi, final_spectra)
```

You can refer to the [examples](tree/main/examples) folder for a complete and commented example.

## Documentation

### XspecTableModelAdditive class

Use `XspecTableModelAdditive` class to create an additive table model.

```
class XspecTableModelAdditive(file_name, model_name, energies, params, redshift=False)
```

Creates the class instance and opens the FITS file from the filesystem if it exists already or creates a new one.

**file_name**  
The path to the FITS file that will be created in the filesystem.  
**model_name**  
The name of the model (letters only, 12 characters max).  
**energies**  
Array of energies to consider for spectra in [keV].  
**params**  
Array of parameters of the model. Each parameter is a tupe with 4 items:
* **name** - name of the parameter (letters only, 12 characters max)
* **grid** - array of parameter values (in increasing order).
* **logarithmic** - True/False flag saying if Xspec shall interpolate in the parameter values linearly (False) or logarithmicaly (True)
* **frozen** - True/False flag saying if Xspec shall initially set this parameter as frozen 
Example: `par1 = ('mass', [10,20,30,40,50], False, False)`
**redshift**
If `redshift` parameter shall be added  by Xspec to the model (boolean). The `redshift` parameter will shift the model in energy space and divide by (1+z) factor.

  Do not include a normalization parameter, it will be added by Xspec automaticaly.

Example:
```python
energies = np.geomspace(1e-2, 1e+2, 100)
param1 = ('alpha', np.linspace(-5,+5,20), False, False)
param1 = ('beta', np.geomspace(1e-1,1e+1, 50), True, False)
model = XspecTableModelAdditive('mymodel.fits', 'mymodel', energies, [param1,param2], False)
```
<br>

```
def XspecTableModelAdditive.generator()
```
Gives an iterator that loops over all combinations of parameter values and allows to provide a spectrum for each row of the spectral table. The iterator returns a tuple with 4 items:
* **index** - index of the row in the spectral table (shall be passed to `write()`)
* **param_values** - array of parameter values for the current spectral row (values are in the same order in which parameters have been passed to the class constructor)
* **param_indexes** - array of parameter indexes for the current spectral row (not really needed, but provided for completeness; you can use the index to get the parameter value from the parameter value grid)
* **energies** - array of energies in [keV] (a copy of the energy grid passed to the class constructor)

The iterator skips any rows that have the spectra filled already. In that way, if the FITS file have existed before, only the missing spectra are computed and so the script allows for a recovery from an interrupted run. On the other hand, if you want to start over, you need to remove the existing FITS file before starting the script.

Example:
```python
model = XspecTableModelAdditive(...)
for g in fits.generator():
    index, param_values, param_indexes, energies = g
    Iv = spectrum(energies, param_values)
    model.write(index, Iv, False)
#end if
```
<br>

```
def XspecTableModelAdditive.write(index, spectrum, flush=False)
```
Write a single spectrum to the table. 
**index**  
Row index of the spectrum (given by the generator).  
**spectrum**  
Energy spectrum (specific flux) in [erg/s/cm2/keV] given at each point of the energy grid.  
**flush**  
If True, the model table is saved to the file system after the spectrum is written.

**Note**: XSPEC requires the table model to contain spectra in units of photons/cm/s (photon spectrum integrated over the energy bin). The spectrum that is passed to `write()` method, however, must be an energy spectrum (specific flux) given at each energy point (not integrated). The integration and conversion to photon spectrum is done inside the function.
<br>

```
def XspecTableModelAdditive.save()
```
Save the content of the FITS to the filesystem.



# torus_integrator
