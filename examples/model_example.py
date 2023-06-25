#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 1, 2023

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
    
    one_table = tables_directory + '/stokes_I_UNPOL.fits'
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
    saved_mui = []
    saved_mue = []
    saved_Phi = []    
    for r, row in enumerate(rows):
        # only the lowest xi (most neutral) gets stored
        if int(row[1]) == 1 and abs(float(row[0]) - Gamma) < 0.001:
            collect.append(r)
            if round(float(row[2]),1) not in saved_mui:
                saved_mui.append(round(float(row[2]),1))
            if float(row[3]) not in saved_Phi:
                saved_Phi.append(float(row[3]))
            if round(float(row[4]),3) not in saved_mue:
                saved_mue.append(round(float(row[4]),3))
    
    return (ener_lo, ener_hi), which_indices[0], which_indices[-1], \
                (saved_mui, saved_mue, saved_Phi), collect
        
def load_tables(tables_directory, name_IQU, first_ind, last_ind, collect):
    # a custom routine to load reflection tables in which we would like to 
    # interpolate
    # the order of parameters is linked to coefficients order
    # in the interpolation routine
    
    three_pols = []    
    # this needs to be done in the corresponding order in
    # interpolation_incident() routine in torus_integrator.py
    for origp in ['UNPOL','HRPOL','45DEG']:
        print('Primary polarization: '+ origp)
        one_table = tables_directory+'/stokes_'+name_IQU+'_'+origp+'.fits'
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
                        # arbitrary normalization factor
                        newrow.append(row[e]/10.**14.)                
                spectra.append(newrow)        
        three_pols.append(spectra)
        
    # rearrange
    loaded_spectra = []
    for s in range(len(three_pols[0])):
        loaded_spectra.append([three_pols[0][s],three_pols[1][s], \
                                   three_pols[2][s]])        
    
    return loaded_spectra


saving_directory = './model_tables'
tables_directory = '.'

# linear binning, not dynamic
N_u = 10 # across (90,270) degrees, the other half is symmetrically added
N_v = 20 # between the shadow line and the equatorial plane,
         # i.e. 180°-Theta <= v <= 180°

# primary polarization states "(name, pol. frac., pol. ang.)" to be computed
PPs = [('UNPOL', 0., 0.), ('PERP100', 1., np.pi/2.), \
               ('45DEG100', 1., np.pi/4.)]

# mu_e emission inclination cosines from the pole to be computed
ms = ['0.025','0.075','0.125','0.175','0.225','0.275','0.325','0.375', \
      '0.425','0.475','0.525','0.575','0.625','0.675','0.725','0.775', \
          '0.825','0.875','0.925','0.975']

# r_in parameters, the torus inner radii (arbitrary units), to be computed
# in the current version this does not impact the results at all
inner_radii = ['0.05']

# Theta half-opening angles to be computed from the pole in degrees
opening_angles = ['25','30','35','40','45','50','55','60','65','70','75',\
                  '80','85']

# which Stokes parameters to compute - if polarization, then specify 
# both 'Q' and 'U', even though Us will be zero in the end due to symmetry
names_IQUs = ['I','Q','U']

# Gammas primary power-law indices to be computed (need to be exactly as in 
# the loaded local tables)
Gammas = ['1.2','1.6','2.0','2.4','3.0']

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
    energies, first_ind, last_ind, parameters, collect \
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
            # my first model
            TM = TorusModel(saving_directory, energies, parameters, \
                                all_spectra, Theta_input, r_in_input, N_u, \
                                N_v, names_IQUs, PPs, ms, Gamma)
            for g in TM.generator():
                name, ener_lo, ener_hi, final_spectra = g
                # save to the desired directory
                TM.save_ascii(name, ener_lo, ener_hi, final_spectra)
