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

### TorusModel class

Use `TorusModel` class to create a spectro-polarimetric reflection table model.

```
class TorusModel(saving_directory, energies, parameters, all_spectra, \
                         Theta_input, r_in_input, N_u, N_v, IQUs, primpols, \
                         mues, Gamma)
```
Stores an ASCII torus model for these user-defined values. Energy binning is expected to be loaded from one sample local reflection table.

* **saving_directory**
path to the directory where to save the results
* **energies**
a tuple of (e_low, e_high), i.e. lower, upper bin boundaries, each being a list containing floats of energy values, as they appear in the local reflection tables
* **parameters**
a tuple of (saved_mui, saved_mue, saved_Phi), each being a list containing floats of local reflection angles, as they appear in the local reflection tables
* **all_spectra**
a list of the stored Stokes parameters, each being a list of energy-dependent values in ['UNPOL','HRPOL','45DEG'] sublist for each primary polarization state, as they appear in the local reflection tables 
* **Theta_input**
a string of half-opening angle from the pole in degrees
* **r_in_input**
a string of inner radius of the circular torus in arbitrary units
* **N_u**
int number of points tried in u direction in linear binning across 180 degrees between 90° and 270° (the other symmetric half-space is added)
* **N_v**
int number of points tried in v direction in linear binning between the shadow line and equatorial plane (i.e. 180° - Theta <= v <= 180°)
* **IQUs**
a list of spectra to be computed, i.e. their names in strings, as they appear in the local reflection tables
* **primpols**
a list of arbitrary primary polarizations to be computed, i.e. tuples containing (name string, p0 float, chi0 float) on which we use the S-formula
* **mues**
a list of cosines of observer's inclinations from the pole to be computed, i.e. strings of any numbers between 0 and 1
* **Gamma**
the power-law index to be computed for, i.e. a string as it appears in the local reflection tables

### LocalPoint class

Use `LocalPoint` class to make all computations in the local reflection frame on the toroidal inner walls.
```
class LocalPoint(u_point, v_point)
```

Makes the angular computations and table interpolations required at each illuminated point of the torus surface.
        
* **u_point**
a float of u in radians defining a point on the torus surface, typically u_mid of one bin
* **v_point**
a float of v in radians defining a point on the torus surface, typically v_mid of one bin
