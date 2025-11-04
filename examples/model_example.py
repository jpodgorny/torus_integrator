#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2, 2025

@author: Jakub Podgorny, jakub.podgorny@asu.cas.cz
"""
import sys
from astropy.io import fits
import numpy as np
import time

# add the path to torus_integrator.py and import the class from the module
sys.path.append('..')
from torus_integrator import TorusModel
    
def get_energies_and_parameters(tables_directory, Gamma, e_min, e_max):
    # get energy and parameter values in keV from any local reflection file
    # the order of parameters is linked to coefficients order
    # in the interpolation routine
    
    one_table = tables_directory + '/stokes_unified_I_UNPOL.fits'
    hdul = fits.open(one_table)
    
    # get energy values
    extname = 'ENERGIES'
    colnames = hdul[extname].columns.names
    for c in range(len(colnames)):
        rows = []
        for i in range(len(hdul[extname].data)):
            onerow = []
            for j in range(len(hdul[extname].data[i])):
                if j == c:
                    if hasattr(hdul[extname].data[i][j], "__len__") and \
                                not type(hdul[extname].data[i][j]) == str:
                        for val in hdul[extname].data[i][j]:
                            onerow.append(val)
                    else:
                        onerow.append(hdul[extname].data[i][j])
            rows.append(onerow)
        if c == 0:
            which_indices = []
            ener_lo = []
            for r, row in enumerate(rows):
                if e_min <= row[0] <= e_max:
                    ener_lo.append(row[0])
                    which_indices.append(r)
        elif c == 1:
            ener_hi = []
            for r, row in enumerate(rows):
                if r in which_indices:
                    ener_hi.append(row[0])
    
    # get parameter values
    extname = 'SPECTRA'
    c = 0
    rows = []
    for i in range(len(hdul[extname].data)):
        if i % 100000 == 0:
            print(i,'/',len(hdul[extname].data))
        onerow = []
        for j in range(len(hdul[extname].data[i])):
            if j == c:
                if hasattr(hdul[extname].data[i][j], "__len__") and \
                            not type(hdul[extname].data[i][j]) == str:
                    for val in hdul[extname].data[i][j]:
                        onerow.append(val)
                else:
                    onerow.append(hdul[extname].data[i][j])
        rows.append(onerow)
    
    # prepare the lists
    collect = []
    saved_xi = []
    saved_mui = []
    saved_mue = []
    saved_Phi = []    
    for r, row in enumerate(rows):
        if abs(float(row[0]) - Gamma) < 0.001:
            collect.append(r)
            if round(float(row[1]),0) not in saved_xi:
                saved_xi.append(round(float(row[1]),0))
            if round(float(row[2]),1) not in saved_mui:
                saved_mui.append(round(float(row[2]),1))
            if float(row[3]) not in saved_Phi:
                saved_Phi.append(float(row[3]))
            if round(float(row[4]),3) not in saved_mue:
                saved_mue.append(round(float(row[4]),3))   
    
    return (ener_lo, ener_hi), which_indices[0], which_indices[-1], \
                (saved_xi, saved_mui, saved_mue, saved_Phi), collect
        
def load_tables(tables_directory, name_IQU, first_ind, last_ind, collect, Gamma, \
                    xi0_code, e_min, e_max):
    # a custom routine to load reflection tables in which we would like to
    # interpolate
    # the order of parameters is linked to coefficients order
    # in the interpolation routine

    loaded_spectra = []
    loaded_spectra_neutral = []

    if xi0_code == 2 or xi0_code == 3:
        # partially ionized tables
        three_pols = []
        # this needs to be done in the corresponding order in
        # interpolation_incident() routine in torus_integrator.py
        for origp in ['UNPOL','VRPOL','45DEG']:
            print('Primary polarization: '+ origp)
            one_table = tables_directory+'/stokes_unified_'+name_IQU+'_'+origp+'.fits'
            hdul = fits.open(one_table)
            extname = 'SPECTRA'
            c = 1 # spectra, not parameter values
            rows = []
            for i in range(len(hdul[extname].data)):
                if i % 100000 == 0:
                    print(i,'/',len(hdul[extname].data))
                onerow = []
                for j in range(len(hdul[extname].data[i])):
                    if j == c:
                        if hasattr(hdul[extname].data[i][j], "__len__") and \
                                    not type(hdul[extname].data[i][j]) == str:
                            for val in hdul[extname].data[i][j]:
                                onerow.append(val)
                        else:
                            onerow.append(hdul[extname].data[i][j])
                rows.append(onerow)

            # save spectra
            spectra = []
            for r, row in enumerate(rows):
                if r in collect:
                    newrow = []
                    for e in range(len(row)):
                        if first_ind <= e <= last_ind:
                            # arbitrary normalization factor,
                            # if changed, change for imaging normalization
                            # inside torus_integrator.py as "K_factor" variable
                            newrow.append(row[e]/10.**14.)
                    spectra.append(newrow)
            three_pols.append(spectra)

        # rearrange
        for s in range(len(three_pols[0])):
            loaded_spectra.append([three_pols[0][s],three_pols[1][s], \
                                    three_pols[2][s]])

    if xi0_code == 1 or xi0_code == 3:
        one_table = tables_directory+'/stokes_tables_neutral_'+name_IQU+'_UNPOL.fits'
        hdul = fits.open(one_table)

        # get energy values
        extname = 'ENERGIES'
        colnames = hdul[extname].columns.names
        for c in range(len(colnames)):
            rows = []
            for i in range(len(hdul[extname].data)):
                onerow = []
                for j in range(len(hdul[extname].data[i])):
                    if j == c:
                        if hasattr(hdul[extname].data[i][j], "__len__") and \
                                    not type(hdul[extname].data[i][j]) == str:
                            for val in hdul[extname].data[i][j]:
                                onerow.append(val)
                        else:
                            onerow.append(hdul[extname].data[i][j])
                rows.append(onerow)
            if c == 0:
                which_indices = []
                ener_lo = []
                for r, row in enumerate(rows):
                    if e_min <= row[0] <= e_max:
                        ener_lo.append(row[0])
                        which_indices.append(r)
            elif c == 1:
                ener_hi = []
                for r, row in enumerate(rows):
                    if r in which_indices:
                        ener_hi.append(row[0])

        # get parameter values
        extname = 'SPECTRA'
        c = 0
        rows = []
        for i in range(len(hdul[extname].data)):
            if i % 100000 == 0:
                print(i,'/',len(hdul[extname].data))
            onerow = []
            for j in range(len(hdul[extname].data[i])):
                if j == c:
                    if hasattr(hdul[extname].data[i][j], "__len__") and \
                                not type(hdul[extname].data[i][j]) == str:
                        for val in hdul[extname].data[i][j]:
                            onerow.append(val)
                    else:
                        onerow.append(hdul[extname].data[i][j])
            rows.append(onerow)

        # prepare the lists
        collecth = []
        saved_mui = []
        saved_mue = []
        saved_Phi = []
        for r, row in enumerate(rows):
            if abs(float(row[0]) - Gamma) < 0.001:
                collecth.append(r)
                if round(float(row[2]),1) not in saved_mui:
                    saved_mui.append(round(float(row[2]),1))
                if float(row[3]) not in saved_Phi:
                    saved_Phi.append(float(row[3]))
                if round(float(row[4]),3) not in saved_mue:
                    saved_mue.append(round(float(row[4]),3))

        # neutral tables
        three_pols = []
        # this needs to be done in the corresponding order in
        # interpolation_incident() routine in torus_integrator.py
        for origp in ['UNPOL','VRPOL','45DEG']:
            print('Primary polarization: '+ origp)
            one_table = tables_directory+'/stokes_tables_neutral_'+name_IQU+'_'+origp+'.fits'
            hdul = fits.open(one_table)
            extname = 'SPECTRA'
            c = 1 # spectra, not parameter values
            rows = []
            for i in range(len(hdul[extname].data)):
                if i % 100000 == 0:
                    print(i,'/',len(hdul[extname].data))
                onerow = []
                for j in range(len(hdul[extname].data[i])):
                    if j == c:
                        if hasattr(hdul[extname].data[i][j], "__len__") and \
                                    not type(hdul[extname].data[i][j]) == str:
                            for val in hdul[extname].data[i][j]:
                                onerow.append(val)
                        else:
                            onerow.append(hdul[extname].data[i][j])
                rows.append(onerow)

            # save spectra
            spectra = []
            for r, row in enumerate(rows):
                if r in collecth:
                    newrow = []
                    for e in range(len(row)):
                        if first_ind <= e <= last_ind:
                            # arbitrary normalization factor,
                            # if changed, change for imaging normalization
                            # inside torus_integrator.py as "K_factor" variable
                            newrow.append(row[e]/10.**14.)
                    spectra.append(newrow)
            three_pols.append(spectra)

        # rearrange
        for s in range(len(three_pols[0])):
            loaded_spectra_neutral.append([three_pols[0][s],three_pols[1][s], \
                                    three_pols[2][s]])

    return loaded_spectra, loaded_spectra_neutral

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
# avoid loading unnecessary tables
has_zero = any(item in ("0", "0.") for item in xi0s)
has_pos  = any(float(item) > 0.0 for item in xi0s)
if not has_zero and not has_pos:
    # do not load any tables if only analytical approximations are required
    xi0_code = 0        # no "0"/"0." and no positive numbers
elif has_zero and not has_pos:
    # load only neutral tables if ionized are not needed
    xi0_code = 1        # "0"/"0." present, but no positive numbers
elif has_pos and not has_zero:
    # load only ionized tables if neutral are not needed
    xi0_code = 2        # "0"/"0." not present, but positive numbers
else:
    # load both neutral and ionized tables
    xi0_code = 3        # "0"/"0." present and positive present
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
                                      last_ind, float(Gamma), xi0_code, \
                                                  E_min, E_max)
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
