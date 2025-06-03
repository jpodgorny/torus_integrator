# Torus reflection integrator

Create simple spectro-polarimetric reflection models off a torus, cone and bowl with the help of this Python class.

## Purpose

This python module takes local spectro-polarimetric reflection tables from FITS files (described in Podgorný et al. 2022, available at https://doi.org/10.6084/m9.figshare.16726207, version 2, or their purely neutral version described in Podgorný 2025, available at https://doi.org/10.6084/m9.figshare.29217854, version 1). It puts a source of X-ray power-law in the center with energy binning, a power-law index Gamma or black-body temperature, and primary polarization according to the local reflection tables. The user defines a range of inner radii, half-opening angles, skews, ionization profiles, and observer's inclinations, for which the final ASCII spectro-polarimetric tables are to be computed. Then the routine takes into account the visible and illuminated part of the inner walls of a toroidal structure and computes the integrated view for a distant observer, seeing a single reflection. Self-irradiation is not taken into account. The output tables show normalized Stokes parameters I, Q, U dependent on energy, named according to the remaining model parameters and their values.

For any issues regarding the use, or bugs spotted, please, contact J. Podgorný at
[jakub.podgorny@asu.cas.cz](mailto:jakub.podgorny@asu.cas.cz) or M. Dovčiak
[michal.dovciak@asu.cas.cz](mailto:michal.dovciak@asu.cas.cz).

## Limitations

* the routines need to be adapted to take into account different reflection tables

## Dependencies

The module uses [AstroPy](https://www.astropy.org/) for reading the FITS files and storing the ASCII files. It also requires the "visibility_line.txt" and "visibility_line_c.txt" to be in the local directory. See the routine headers for all required libraries.

## References

Podgorný J, Dovčiak M, Marin F, Goosmann RW and Różańska A (2022)
_Spectral and polarization properties of reflected X-ray emission from black hole accretion discs_
[MNRAS, 510, pp.4723-4735](https://doi.org/10.1093/mnras/stab3714)
[[arXiv:2201.07494](https://arxiv.org/abs/2201.07494)]

Podgorný J, Dovčiak M, Marin F (2024)
_Simple numerical X-ray polarization models of reflecting axially symmetric structures around accreting compact objects_
[MNRAS, 530, pp.2608-2626](https://doi.org/10.1093/mnras/stae1009)
[[arXiv:2310.15647](https://arxiv.org/abs/2310.15647)]

Podgorný J (2025)
_Shape and ionization of equatorial matter near compact objects from X-ray polarization reflection signatures_
[[arXiv:2506.01798](https://arxiv.org/abs/2506.01798)]

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

# choose geometry type
# 'torus' = classical elliptical concave torus,
# 'cone' = double cone with straight walls,
# 'bowl' = elliptical convex double cone, i.e. inverted elliptical torus
geometry_list = ['cone','bowl']

# choose source type
# 'iso', a(mu) = 1, isotropic source
# 'csource', a(mu) = 2*mu, a cosine source
# 'slabcorona', a(mu) from Eq.(26) in Nitindala et al. (2025), a slab corona example
# - provides polarization prescription too
# 'edisc', a(mu) from Eq. (27) in Nitndala et al. (2025), an electron-scattering dominated disc
# - provides polarization prescription too (Eq. (41) in Viironen & Poutanen (2004))
inctype_list = ['iso','csource','slabcorona','edisc']

# whether to produce text files with primary radiation (non-zero for i < Theta)
produce_primary = True

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
         # must be even for cone or bowl

# isotropic primary polarization states "(name, pol. frac., pol. ang.)" to be computed
# inctype = 'slabcorona' and inctype = 'edisc' have given anisotropic primary polarization state,
# then this value can be any
PPs = [('UNPOL', 0., 0.), ('PARA100', 1., 0.), \
               ('45DEG100', 1., np.pi/4.)]

# 0 < mu_e < 1 emission inclination cosines from the pole to be computed
ms = ['0.025','0.075','0.125','0.175','0.225','0.275','0.325','0.375', \
      '0.425','0.475','0.525','0.575','0.625','0.675','0.725','0.775', \
          '0.825','0.875','0.925','0.975']

# r_in parameters, the torus inner radii (arbitrary units), to be computed
# in the current version this does not impact the results at all
inner_radii = ['1.']

# if float(xi_0) > 0 parameters to be computed via interpolation of the local reflection tables
# if xi0 = '0.', then STOKES neutral spectro-polarimetric tables are used as if xi0 = 1.
xi0s = ['10.','100.','1000.','10000.']

# beta parameters to be computed
betas = ['0.','2.']

# rho parameters, i.e. the distance between the z-axis and the upper or lower
# end of the illuminated part of the inner walls, put 'c' for a circular torus
# in the 'torus' geometry
rhos = ['1.','5.','10.','50.','100.']

# 1° <= Theta <= 89° half-opening angles to compute from the pole in degrees
# for cone and elcone
opening_angles_degrees = ['25','30','35','40','45','50','55','60','65','70', \
                          '75','80','85']
# 0 <= Theta' <= 1 rescaled half-opening angles to be transformed to real
# Theta, depending on inclination
# for torus
opening_angles_transformed = ['0.00','0.05','0.10','0.15','0.20','0.25', \
                              '0.30','0.35','0.40','0.45','0.50','0.55', \
                              '0.60','0.65','0.70','0.75','0.80','0.85', \
                              '0.90','0.95','1.00']
#opening_angles = opening_angles_transformed
opening_angles = opening_angles_degrees

# which Stokes parameters to compute - if polarization, then specify
# both 'Q' and 'U', even though Us will be zero in the end due to symmetry
names_IQUs = ['I','Q','U']

# Gammas primary power-law indices, or T_BB primary temperature for single-temperature blackbody emission,
# to be computed (need to be exactly as in the loaded local tables)
# this is named always in the output as Gamma, even if it has a meaning of T_BB
Gammas_or_TBBs = ['1.4','1.6','1.8','2.0','2.2','2.4','2.6','2.8','3.0']

# Imaging: leave this list empty, if no imaging is needed
# list of tuples of parameters, for which to provide images and image data,
# each tuple will contain: (name of prim. pol., mu_e, Theta, Gamma or T_BB, r_in, rho, xi0, beta)
image_list = []
'''
image_list = [('UNPOL','0.425','40','2.0','1.','10.','10.','2.'), \
              ('UNPOL','0.875','40','2.0','1.','10.','10.','2.'), \
              ('UNPOL','0.425','70','2.0','1.','10.','10.','2.'), \
              ('UNPOL','0.875','70','2.0','1.','10.','10.','2.'), \
                  ('UNPOL','0.425','40','2.0','1.','100.','10.','2.'), \
                                ('UNPOL','0.875','40','2.0','1.','100.','10.','2.'), \
                                ('UNPOL','0.425','70','2.0','1.','100.','10.','2.'), \
                                ('UNPOL','0.875','70','2.0','1.','100.','10.','2.')]
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
for Gamma in Gammas_or_TBBs:
    print('beginning to compute Gamma or T_BB: ', Gamma)

    # custom routines to get energies, parameters and spectra for
    # interpolation
    E_min = 0.1 # for low bin edge
    E_max = 100. # for low bin edge
    # use any local reflection table to get the energy values and parameter
    # values
    print('... loading energy values and parameter values')
    energies, first_ind, last_ind, parameters, collect \
            = get_energies_and_parameters(tables_directory, float(Gamma), \
                                                  E_min, E_max)
    # get the relevant spectra per Gamma
    all_spectra = []
    all_spectra_neutral = []
    for name_iqu in names_IQUs:
        print('... loading Stokes '+name_iqu)
        one_iqu, one_iqu_neutral = load_tables(tables_directory, name_iqu, first_ind, \
                                      last_ind, collect)
        all_spectra.append(one_iqu)
        all_spectra_neutral.append(one_iqu_neutral)

    # produce the table models
    for geometry in geometry_list:
        for inctype in inctype_list:
            for r_in_input in inner_radii:
                for xi0_input in xi0s:
                    for beta_input in betas:
                        for rho_input in rhos:
                            for Theta_input in opening_angles:
                                time_now = int(time.time())
                                print('currently computing reflection for r_in: ',r_in_input,\
                                      'and Theta: ', \
                                          Theta_input, 'after ', \
                                          round(abs(last_time-time_now)/60./60.,3), 'hours')
                                for mue in ms:
                                    # let's compute a model for these parameter values
                                    TM = TorusModel(saving_directory, energies, parameters,\
                                                        all_spectra, all_spectra_neutral, Theta_input, r_in_input,\
                                                        N_u, N_v, names_IQUs, PPs, mue, Gamma,\
                                                        below_equator, image_list, \
                                                        image_resolution, image_energy_ranges,\
                                                        image_limits, geometry, rho_input, xi0_input, beta_input, \
                                                        inctype, produce_primary)
                                    for g in TM.generator():
                                        name, ener_lo, ener_hi, final_spectra, final_spectra_prim = g
                                        # save to the desired directory
                                        TM.save_ascii(name, ener_lo, ener_hi, final_spectra)
                                        if produce_primary == True:
                                            TM.save_ascii_prim(name, ener_lo, ener_hi, final_spectra_prim)
```

You can refer to the [examples](tree/main/examples) folder for a complete and commented example. In the same folder you need to place the local reflection tables downloaded from: https://doi.org/10.6084/m9.figshare.16726207 (version 2) and https://doi.org/10.6084/m9.figshare.29217854 (version 1).

## Documentation

### TorusModel class

Use `TorusModel` class to create a spectro-polarimetric reflection table model.

```
class TorusModel(saving_directory, energies, parameters, all_spectra, all_spectra_neutral, \
                         Theta_input, r_in_input, N_u, N_v, IQUs, primpols, \
                         mue, Gamma, below_equator, image_list, \
                         image_resolution, image_energy_ranges, image_limits, \
                         geometry, rho_input, xi0_input, beta_input, inctype, produce_primary)
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
a tuple of (saved_xi, saved_mui, saved_mue, saved_Phi), each
being a list containing floats of ionization and local reflection
angles, as they appear in the local reflection tables
* **all_spectra, all_spectra_neutral**
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
* **geometry**
a string that identifies the geometry to be computed
* **rho_input**
a string containing the value of the rho geometrical parameter
* **xi0_input**
a string containing the value of the xi0 ionization normalization parameter
* **beta_input**
a string containing the value of the beta ionization profile parameter
* **inctype**
a string that identifies the type of incident emission
* **produce_primary**
a boolean whether to produce a results text file with
primary radiation spectrum at r_in


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
