# Torus reflection integrator

Create simple spectro-polarimetric reflection models off a torus with the help of this Python class.

## Purpose

This python module takes local spectro-polarimetric reflection tables from FITS files (described in Podgorný et al. 2022, available at https://doi.org/10.6084/m9.figshare.16726207, version 2). It puts a source of X-ray power-law in the center with energy binning, a power-law index Gamma, and primary polarization according to the local reflection tables. The user defines a range of inner radii, torus half-opening angles and observer's inclinations, for which the final ASCII spectro-polarimetric tables are to be computed. Then the routine takes into account the visible and illuminated part of the inner walls of a pure circular torus and computes the integrated view for a distant observer, seeing a single reflection. Direct radiation or self-irradiation are not taken into account. The output tables show renormalized Stokes parameters I, Q, U dependent on energy, named according to the remaining model parameters and their values.

For any issues regarding the use of xsstokes_torus, please, contact J. Podgorný at 
[jakub.podgorny@asu.cas.cz](mailto:jakub.podgorny@asu.cas.cz) or M. Dovčiak
[michal.dovciak@asu.cas.cz](mailto:michal.dovciak@asu.cas.cz).

## Limitations

* only reflection from a pure circular torus can be computed at the moment
* the routines need to be adapted to take into account different reflection tables

## Dependencies

The module uses [AstroPy](https://www.astropy.org/) for reading the FITS files and storing the ASCII files. It also requires the "visibility_line.txt" to be in the local directory. See the routine headers for all required libraries.

## References

Podgorný J, Dovčiak M, Marin F, Goosmann RW and Różańska A (2022)
_Spectral and polarization properties of reflected X-ray emission from black hole accretion discs_
[MNRAS, 510, pp.4723-4735](https://doi.org/10.1093/mnras/stab3714)
[[arXiv:2201.07494](https://arxiv.org/abs/2201.07494)]

Podgorný J, Dovčiak M, Marin F (2024)
_Simple numerical X-ray polarization models of reflecting axially symmetric structures around accreting compact objects_
[MNRAS, 530, pp.2608-2626](https://doi.org/10.1093/mnras/stae1009)
[[arXiv:2310.15647](https://arxiv.org/abs/2310.15647)]

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

# if producing images, create also the "images" and "images_data" directories
# inside the saving directory, in order to store the image there
saving_directory = './model_tables'
tables_directory = '.'

# whether to include the visible region below the torus equator:
# B = 1 is True (accounted for), B = 0 is False (not accounted for)
below_equator = True

# linear binning, not dynamic
N_u = 50 # across (90,270) degrees, the other half is symmetrically added
N_v = 80 # between the upper shadow line and the equatorial plane or lower
         # shadow line, i.e.
         # 180°-Theta <= v <= 180°, if below_equator == False;
         # 180°-Theta <= v <= 180°+Theta, if below_equator == True

# primary polarization states "(name, pol. frac., pol. ang.)" to be computed
PPs = [('UNPOL', 0., 0.), ('PERP100', 1., np.pi/2.), \
               ('45DEG100', 1., np.pi/4.)]

# 0 < mu_e < 1 emission inclination cosines from the pole to be computed
ms = ['0.025','0.075','0.125','0.175','0.225','0.275','0.325','0.375', \
      '0.425','0.475','0.525','0.575','0.625','0.675','0.725','0.775', \
          '0.825','0.875','0.925','0.975']

# r_in parameters, the torus inner radii (arbitrary units), to be computed
# in the current version this does not impact the results at all
inner_radii = ['0.05']

# 1° <= Theta <= 89° half-opening angles to compute from the pole in degrees
opening_angles_degrees = ['25','30','35','40','45','50','55','60','65','70', \
                          '75','80','85']
# 0 <= Theta' <= 1 rescaled half-opening angles to be transformed to real 
# Theta, depending on inclination
opening_angles_transformed = ['0.00','0.05','0.10','0.15','0.20','0.25', \
                              '0.30','0.35','0.40','0.45','0.50','0.55', \
                              '0.60','0.65','0.70','0.75','0.80','0.85', \
                              '0.90','0.95','1.00']
opening_angles = opening_angles_transformed

# which Stokes parameters to compute - if polarization, then specify 
# both 'Q' and 'U', even though Us will be zero in the end due to symmetry
names_IQUs = ['I','Q','U']

# Gammas primary power-law indices to be computed (need to be exactly as in 
# the loaded local tables)
Gammas = ['2.0']

# Imaging: leave this list empty, if no imaging is needed
# list of tuples of parameters, for which to provide images and image data,
# each tuple will contain: (name of prim. pol., mu_e, Theta, Gamma)
image_list = []
'''
image_list = [('UNPOL','0.425','40','2.0'), \
              ('UNPOL','0.925','40','2.0'), \
              ('UNPOL','0.425','70','2.0'), \
              ('UNPOL','0.925','70','2.0')]
'''

# imaging energy bands, provide a list of tuples of minimum and maximum
# energy in keV over which to integrate and provide a final image
image_energy_ranges = [(3.5,6.),(30.,60.)]

# imaging resolution, number of pixels > 1 linearly spread in X and Y to appear
# in the image data (recommended value 25 produces 25x25 picture)
image_resolution = 25

# imaging lower and upper limits on the colorbar axis that plots pF/F_*
image_limits = [10.**(-8.),10.**(-3.)]

last_time = int(time.time())
for Gamma in Gammas:
    print('beginning to compute Gamma: ', Gamma)
    
    # custom routines to get energies, parameters and spectra for
    # interpolation    
    E_min = 1. # for low bin edge
    E_max = 100. # for low bin edge
    # use any local reflection table to get the energy values and parameter 
    # values
    print('... loading energy values and parameter values')
    energies, first_ind, last_ind, parameters, collect, xi \
            = get_energies_and_parameters(tables_directory, float(Gamma), \
                                                  E_min, E_max)
    # get the relevant spectra per Gamma
    all_spectra = []
    for name_iqu in names_IQUs:
        print('... loading Stokes '+name_iqu)
        one_iqu = load_tables(tables_directory, name_iqu, first_ind, \
                                      last_ind, collect)
        all_spectra.append(one_iqu)
        
    # produce the table models
    for r_in_input in inner_radii:
        for Theta_input in opening_angles:
            time_now = int(time.time())
            print('currently computing reflection for r_in: ',r_in_input, \
                  'and Theta: ', \
                      Theta_input, 'after ', \
                      round(abs(last_time-time_now)/60./60.,3), 'hours')
            for mue in ms: 
                
                # my first model
                TM = TorusModel(saving_directory, energies, parameters, \
                                    all_spectra, Theta_input, r_in_input, \
                                    N_u, N_v, names_IQUs, PPs, mue, Gamma, \
                                    xi, below_equator, image_list, \
                                    image_resolution, image_energy_ranges, \
                                    image_limits)
                for g in TM.generator():
                    name, ener_lo, ener_hi, final_spectra = g
                    # save to the desired directory
                    TM.save_ascii(name, ener_lo, ener_hi, final_spectra)
```

You can refer to the [examples](tree/main/examples) folder for a complete and commented example. In the same folder you need to place the local reflection tables downloaded from: https://doi.org/10.6084/m9.figshare.16726207 (version 2).

## Documentation

### TorusModel class

Use `TorusModel` class to create a spectro-polarimetric reflection table model.

```
class TorusModel(saving_directory, energies, parameters, \
                                    all_spectra, Theta_input, r_in_input, \
                                    N_u, N_v, names_IQUs, PPs, mue, Gamma, \
                                    xi, below_equator, image_list, \
                                    image_resolution, image_energy_ranges, \
                                    image_limits)
```
Stores an ASCII torus model for these user-defined values. Energy binning is expected to be loaded from one sample local reflection table.

* **saving_directory**
path to the directory where to save the results
* **energies**
a tuple of (e_low, e_high), i.e. lower, upper bin
boundaries, each being a list containing floats
of energy values, as they appear in the local
reflection tables
* **parameters**
a tuple of (saved_mui, saved_mue, saved_Phi), each
being a list containing floats of local reflection
angles, as they appear in the local reflection tables
* **all_spectra**
a list of the stored Stokes parameters, each being 
a list of energy-dependent values in ['UNPOL','HRPOL',
'45DEG'] sublist for each primary polarization state,
as they appear in the local reflection tables 
* **Theta_input**
a string of half-opening angle Theta from the pole in
degrees, if greater than 1; a string of transformed
half-opening angle Theta', if lower or equal to 1
* **r_in_input**
a string of inner radius of the circular torus in
arbitrary units
* **N_u**
int number of points tried in u direction in linear binning
across 180 degrees between 90° and 270° (the other
symmetric half-space is added)
* **N_v**
int number of points tried in v direction in linear binning
between the shadow line and equatorial plane
(i.e. 180° - Theta <= v <= 180°)
* **IQUs**
a list of spectra to be computed, i.e. their names in
strings, as they appear in the local reflection tables
* **primpols**
a list of arbitrary primary polarizations to be computed,
i.e. tuples containing (name string, p0 float, 
chi0 float) on which we use the S-formula
* **mue**
a string of cosine of observer's inclinations from the pole
to be computed, i.e. a string of any number
between 0 and 1
* **Gamma**
the power-law index to be computed for, i.e. a string as it
appears in the local reflection tables
* **xi**
the ionization parameter of the reflection tables, i.e.
a float number dependent on Gamma needed for imaging
* **below_equator**
a boolean whether to take into account the visible
area below the torus equator (yes = True)
* **image_list**
a list of tuples, each being a unique parameter
combination, for which an image should be created
* **image_resolution**
an integer that sets the number of pixels in
X and Y dimensions, if any image is created
* **image_energy_ranges**
a list of tuples, each containing a minimum
and maximum energy in keV, defining a range for
which to create the images
* **image_limits**
a list of two floats, i.e. the lower and upper limit
of the colorbar axis on images


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
