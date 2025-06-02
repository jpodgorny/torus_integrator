#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2, 2025

@author: Jakub Podgorny, jakub.podgorny@asu.cas.cz
"""
import sys
from operator import add
from astropy.io import ascii
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter

class TorusModel:
    # class variables need a default value
    inc_t = np.pi/4.
    u_m = np.pi/4.
    v_m = np.pi/4.
    theta_t = np.pi/4.
    R_t = 1.
    a_t = 1.
    b_t = 1.
    r_in_t = 1.
    xi0_t = 1
    xi0_val_t = 1.
    beta_t = 1.
    rho_t = 1.
    geometry_code_t = 0
    inctype_code_t = 0
    IQUs_t = []
    Gamma_t = '2.0'
    
    def __init__(self, saving_directory, energies, parameters, all_spectra, all_spectra_neutral, \
                         Theta_input, r_in_input, N_u, N_v, IQUs, primpols, \
                         mue, Gamma, below_equator, image_list, \
                         image_resolution, image_energy_ranges, image_limits, \
                         geometry, rho_input, xi0_input, beta_input, inctype, produce_primary):
        """
        Store an ASCII torus model for these user-defined values. Energy 
        binning expected to be loaded from one sample local reflection table.
        
        Args:
            saving_directory: path to the directory where to save the results
            energies: a tuple of (e_low, e_high), i.e. lower, upper bin
                        boundaries, each being a list containing floats
                        of energy values, as they appear in the local
                        reflection tables
            parameters: a tuple of (saved_xi, saved_mui, saved_mue, saved_Phi), each
                        being a list containing floats of ionization and local reflection
                        angles, as they appear in the local reflection tables
            all_spectra, all_spectra_neutral: a list of the stored Stokes parameters, each being 
                        a list of energy-dependent values in ['UNPOL','HRPOL',
                        '45DEG'] sublist for each primary polarization state,
                        as they appear in the local reflection tables 
            Theta_input: a string of half-opening angle Theta from the pole in
                        degrees, if greater than 1; a string of transformed
                        half-opening angle Theta', if lower or equal to 1
            r_in_input: a string of inner radius of the circular torus in
                        arbitrary units
            N_u: int number of points tried in u direction in linear binning
                        across 180 degrees between 90° and 270° (the other
                        symmetric half-space is added)
            N_v: int number of points tried in v direction in linear binning
                        between the shadow line and equatorial plane
                        (i.e. 180° - Theta <= v <= 180°)
            IQUs: a list of spectra to be computed, i.e. their names in
                        strings, as they appear in the local reflection tables
            primpols: a list of arbitrary primary polarizations to be computed,
                        i.e. tuples containing (name string, p0 float, 
                        chi0 float) on which we use the S-formula
            mue: a string of cosine of observer's inclinations from the pole
                        to be computed, i.e. a string of any number
                        between 0 and 1
            Gamma: the power-law index to be computed for, i.e. a string as it
                        appears in the local reflection tables
            below_equator: a boolean whether to take into account the visible
                        area below the torus equator (yes = True)
            image_list: a list of tuples, each being a unique parameter
                        combination, for which an image should be created
            image_resolution: an integer that sets the number of pixels in
                        X and Y dimensions, if any image is created
            image_energy_ranges: a list of tuples, each containing a minimum
                        and maximum energy in keV, defining a range for
                        which to create the images
            image_limits: a list of two floats, i.e. the lower and upper limit
                        of the colorbar axis on images
            geometry: a string that identifies the geometry to be computed
            rho_input: a string containing the value of the rho geometrical parameter
            xi0_input: a string containing the value of the xi0 ionization normalization parameter
            beta_input: a string containing the value of the beta ionization profile parameter
            inctype: a string that identifies the type of incident emission
            produce_primary: a boolean whether to produce a results text file with
                        primary radiation spectrum at r_in
        """
        
        self.saving_directory = saving_directory
        self.energies = energies
        self.parameters = parameters
        self.all_spectra = all_spectra
        self.all_spectra_neutral = all_spectra_neutral
        self.Theta_input = Theta_input
        self.r_in_input = r_in_input
        self.N_u = N_u
        self.N_v = N_v
        self.IQUs = IQUs
        self.primpols = primpols
        self.Gamma = Gamma
        self.mue = mue
        self.below_equator = below_equator
        self.image_list = image_list
        self.image_resolution = image_resolution
        self.image_energy_ranges = image_energy_ranges
        self.image_limits = image_limits
        self.geometry = geometry
        self.rho_input = rho_input
        self.xi0_input = xi0_input
        self.beta_input = beta_input
        self.inctype = inctype
        self.produce_primary = produce_primary
        
        self.geometry_code, self.inctype_code, self.xi0, self.xi0_val = self.get_intcodes()
        self.inc_tot = np.arccos(float(self.mue))
        self.Theta = self.convert_Theta()
        self.reverse_ho = np.pi/2. - self.Theta/180.*np.pi 
        self.r_in = float(self.r_in_input)
        self.rho = self.convert_rho()
        self.beta = float(beta_input)
        self.R, self.a, self.b = self.torus_parameters()
        self.u_grid, self.v_grid = self.get_grid()   

    @staticmethod
    def read_table(file_path):
        """
        Reads the text file containing the table.
        The first row contains the x-values, the first column contains the y-values,
        and the remaining cells are z-values.

        Args:
            file_path (str): Path to the text file.
        Returns:
            tuple: x_values, y_values, z_values as numpy arrays.
        """
        data = np.genfromtxt(file_path, delimiter="\t", dtype=str)
        
        # Extract x-values (from the first row, skipping the first cell)
        x_values = np.array(data[0, 1:], dtype=float)
        
        # Extract y-values (from the first column, skipping the first row)
        y_values = np.array(data[1:, 0], dtype=float)
        
        # Extract z-values (remaining part of the table)
        z_values = np.array(data[1:, 1:], dtype=float)
        
        return x_values, y_values, z_values

    @staticmethod
    def compute_a_mu(cosang,itype):
        # compute the a_mu factor for a given source type
        
        if itype == 0:
            a_mu = 1.
        elif itype == 1:
            a_mu = 2.*cosang
        elif itype == 2:
            a_mu = 1.73*cosang*(1.+5.3*cosang- \
                                       0.2*cosang*cosang)/(1.+1.3*cosang+ \
                                                                  4.4*cosang*cosang)
        elif itype == 3:
            a_mu = 2.*cosang*(1.+2.06*cosang)/(1.+2.*2.06/3.)
            
        return a_mu
    
    def interpolate_thetalim(self, file_path):
        """
        Finds the y-value for the given x and z using the data from the text file.

        Args:
            x (float): x-coordinate for interpolation.
            z (float): Target z-value.
            file_path (str): Path to the text file containing the data.

        Returns:
            float: Interpolated y-value.
        """
        
        x = float(self.rho_input)
        z = self.inc_tot/np.pi*180.
        
        # Read the data from the file
        x_values, y_values, z_values = TorusModel.read_table(file_path)

        # Interpolate z-values for the given x across all y-values
        z_interpolator = RegularGridInterpolator((y_values, x_values), z_values)

        z_values_for_x = z_interpolator((y_values, x))  # Interpolate z(y) for this x

        # Tolerant monotonicity check
        if not (np.all(np.diff(z_values_for_x) >= -1e-10) or np.all(np.diff(z_values_for_x) <= 1e-10)):
            raise ValueError("z(y) is not monotonic for this x.")

        # Use interp1d to invert z(y) into y(z)
        interp = interp1d(
            z_values_for_x,
            y_values,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True
        )

        # Compute the corresponding y for the given z
        return float(interp(z))
    
    def get_intcodes(self):
        if self.geometry == 'torus':
            geometry_code = 0
        elif self.geometry == 'bowl':
            geometry_code = 1
        else:
            geometry_code = 2
            
        if self.inctype == 'iso':
            inctype_code = 0
        elif self.inctype == 'csource':
            inctype_code = 1
        elif self.inctype == 'slabcorona':
            inctype_code = 2
        else:
            inctype_code = 3
            
        xi0 = 0
        xi0_val = float(self.xi0_input)

        # Compute the corresponding y for the given z
        return geometry_code, inctype_code, xi0, xi0_val
    
    def convert_Theta(self):
        # convert Theta, if the input needs to be transformed
        
        if 1. < float(self.Theta_input) < 90.:
            # the input is already in degrees
            true_Theta = float(self.Theta_input)
            
        elif 0. <= float(self.Theta_input) <= 1.:
            # the input needs to be transformed
            
            if self.geometry_code == 0 and self.rho_input == 'c':
                vis_Theta = []
                vis_inc = []
                lines = []
                with open('../visibility_line_c.txt', 'rt') as in_file:
                    for line in in_file:
                        lines.append(line.rstrip('\n'))
                    for i in range(len(lines)):  
                        splitted = lines[i].split()
                        vis_Theta.append(float(splitted[0]))
                        vis_inc.append(float(splitted[1]))
                        
                interp_func_x = interp1d(vis_inc, vis_Theta, kind='linear', \
                                         fill_value="extrapolate")
                Theta_limit = interp_func_x(self.inc_tot/np.pi*180.)
            else:
                Theta_limit = self.interpolate_thetalim('../visibility_line.txt')
                
            if Theta_limit >= 90.:
                Theta_limit = 89.9999
            min_Theta = max(0.,Theta_limit)
            max_Theta = 90.
            
            # Theta must be lower than 90 degrees and above the visibility line
            if float(self.Theta_input) == 1.:
                true_Theta = 0.9999*(max_Theta - min_Theta) + min_Theta
            elif float(self.Theta_input) < 0.01:
                true_Theta = 0.01*(max_Theta - min_Theta) \
                                + min_Theta
            else:
                true_Theta = float(self.Theta_input)*(max_Theta - min_Theta) \
                                + min_Theta
        else:
            print("error: input half-opening angle must be between 0. and " \
                  + "90.; if greater than 1, then in degrees; if lower or " \
                  + "equal to 1, then in Theta'")
            sys.exit()
        
        return true_Theta
    
    def convert_rho(self):
        # shift rho by a small amount, so that the torus and bowl are properly
        # defined and check if not circular torus is imposed
        
        if self.rho_input == 'c':
            true_rho = self.r_in/((1.-np.cos(self.Theta/180.*np.pi))*(1.+(1/np.tan(self.Theta/180.*np.pi))**2.))
        else:
            if (self.geometry_code == 1 or self.geometry_code == 0) and \
                        float(self.rho_input) == self.r_in:
                true_rho = float(self.rho_input) + 0.0001*self.r_in
            else:
                true_rho = float(self.rho_input)
        
        return true_rho
    
    def torus_parameters(self):
        # calculate main geometrical parameters of a 'torus' or 'bowl'
        
        if self.geometry_code == 1:
            a = self.rho - self.r_in
            b = self.rho/np.tan(self.Theta/180.*np.pi)
            R = self.r_in
        elif self.geometry_code == 0:
            h = self.rho / np.tan(self.Theta / 180. * np.pi)
            a = self.r_in*(self.rho-self.r_in)/(2.*self.r_in-self.rho)
            b = (-h*a*a/(self.rho-a-self.r_in)/ np.tan(self.Theta/180.*np.pi))**(0.5)
            R = a + self.r_in
        else:
            a = -1.
            b = -1.
            R = -1.
        
        return R, a, b
    
    def get_grid(self):
        # return a list of tuples containing a triplet of u_mid, u_1, u_2 and
        # v_mid, v_1, v_2
        
        # the result is symmetric in the mid meridional plane for u variable,
        # thus we multiply by 2 in the end
        
        u_min = np.pi/2.
        u_max = 3.*np.pi/2.
        delta_u = (u_max - u_min)/float(self.N_u-1)
        
        u_mid = np.linspace(u_min, u_max, num=self.N_u)
        u_low = u_mid - delta_u/2.
        u_high = u_mid + delta_u/2.
        
        # first and last bin are half-size
        u_low[0] = u_mid[0]
        u_mid[0] = u_low[0] + delta_u/4.
        u_high[-1] = u_mid[-1]
        u_mid[-1] = u_high[-1] - delta_u/4.
        
        if self.geometry_code == 2:
            if self.N_v % 2 != 0:
                print('error: for a cone the number of bins in v must be even, '+ \
                      +'because the middle bin at z = 0 cannot be extending through both the top and bottom part')
                sys.exit()
                
            v_max = self.rho/np.tan(self.Theta/180.*np.pi)
            if self.below_equator:
                v_min = - self.rho/np.tan(self.Theta/180.*np.pi)
            else:
                v_min = 0.0001 # to allow PA to be defined
            delta_v = (v_max - v_min)/float(self.N_v-1)
            
            v_mid = np.linspace(v_min, v_max, num=self.N_v)
            v_low = v_mid - delta_v/2.
            v_high = v_mid + delta_v/2.
            
            # first and last bin are half-size
            v_low[0] = v_mid[0]
            v_mid[0] = v_low[0] + delta_v/4.
            v_high[-1] = v_mid[-1]
            v_mid[-1] = v_high[-1] - delta_v/4.
            
        elif self.geometry_code == 1:
            if self.N_v % 2 != 0:
                print('error: for a bowl the number of bins in v must be even, '+ \
                      +'because the middle bin at z = 0 cannot be extending through both the top and bottom part')
                sys.exit()
            v_min1 = 0.
            v_max1 = np.pi/2.-0.0001 # to allow PA to be defined
            if self.below_equator:
                v_min2 = np.pi
                v_max2 = 3.*np.pi/2.-0.0001 # to allow PA to be defined
                # this should be generalized for odd N_v later
                v_mid1 = np.linspace(v_min1, v_max1, num=int(self.N_v/2.))
                v_mid2 = np.linspace(v_min2, v_max2, num=int(self.N_v/2.))
                
                delta_v = (v_max1 - v_min1)/float(self.N_v-1)*2. # doubling the size
                v_low1 = v_mid1 - delta_v/2.
                v_high1 = v_mid1 + delta_v/2.
                v_low2 = v_mid2 - delta_v/2.
                v_high2 = v_mid2 + delta_v/2.
                
                v_low1[0] = v_mid1[0]
                v_high1[-1] = v_mid1[-1]
                v_low2[0] = v_mid2[0]
                v_high2[-1] = v_mid2[-1]
                
                v_mid1[0] = v_low1[0] + delta_v/4.
                v_mid1[-1] = v_high1[-1] - delta_v/4.
                v_mid2[0] = v_low2[0] + delta_v/4.
                v_mid2[-1] = v_high2[-1] - delta_v/4.
                
                v_mid = np.asarray(list(v_mid1) + list(v_mid2))
                v_low = np.asarray(list(v_low1) + list(v_low2))
                v_high = np.asarray(list(v_high1) + list(v_high2))
                
            else:
                v_mid = np.linspace(v_min1, v_max1, num=self.N_v)
                delta_v = (v_max1 - v_min1)/float(self.N_v-1)
            
                v_low = v_mid - delta_v/2.
                v_high = v_mid + delta_v/2.
                
                # first and last bin are half-size
                v_low[0] = v_mid[0]
                v_mid[0] = v_low[0] + delta_v/4.
                v_high[-1] = v_mid[-1]
                v_mid[-1] = v_high[-1] - delta_v/4.
            
        else:
            height = self.rho/(self.b*np.tan(self.Theta/180.*np.pi))
            if abs(height - 1.) < 0.01:
                height = 1.
            elif abs(height + 1.) < 0.01:
                height = -1.
            
            v_min = np.pi-np.arcsin(height)
            if self.below_equator:
                v_max = np.pi+np.arcsin(height)
            else:
                v_max = np.pi-0.0001 # to allow PA to be defined
            delta_v = (v_max - v_min)/float(self.N_v-1)
            
            v_mid = np.linspace(v_min, v_max, num=self.N_v)
            v_low = v_mid - delta_v/2.
            v_high = v_mid + delta_v/2.
            
            # first and last bin are half-size
            v_low[0] = v_mid[0]
            v_mid[0] = v_low[0] + delta_v/4.
            v_high[-1] = v_mid[-1]
            v_mid[-1] = v_high[-1] - delta_v/4.
            
        return list(zip(u_mid, u_low, u_high)), list(zip(v_mid, v_low, v_high))
    
    def name_file(self, pp):
        # how the saved ASCII files should be named
        
        if float(self.Theta_input) > 1.:
            name_Theta = '_Theta' + self.Theta_input
        else:
            name_Theta = '_trTheta' + self.Theta_input
            
        name = self.saving_directory + '/'+ self.geometry + '_' + self.inctype + '_uv' + str(self.N_u) + 'by' + \
                    str(self.N_v) + '_be' + str(int(self.below_equator)) + \
                    name_Theta + '_rho' + self.rho_input + '_rin' + \
                    self.r_in_input + '_xi0' + self.xi0_input + '_beta' + \
                        self.beta_input + '_Gamma' + self.Gamma + '_mue' + \
                    self.mue + '_prim' + pp[0]
        
        return name
    
    def name_image(self, er_vals, pp):
        # how the saved images and image data should be named
        
        if float(self.Theta_input) > 1.:
            name_Theta = '_Theta' + self.Theta_input
        else:
            name_Theta = '_trTheta' + self.Theta_input
        
        image_name = '/image_'+ self.geometry + '_' + self.inctype + '_uv' + str(self.N_u) + 'by' + \
                    str(self.N_v) + '_be' + str(int(self.below_equator)) + \
                    name_Theta + '_rho' + self.rho_input + '_rin' + \
                        self.r_in_input + '_xi0' + self.xi0_input + '_beta' + \
                            self.beta_input + '_Gamma' + \
                    self.Gamma + '_mue' + self.mue + '_prim' + pp[0] + \
                    '_erange' + str(er_vals[0]) + 'to' + str(er_vals[1])
        
        return image_name
    
    def calculate_v_lim(self):
        # calculating the visibility boundary
        
        local_v_limit = - np.arctan( self.b * np.sin(TorusModel.u_m) * \
                            np.tan(self.inc_tot) / self.a ) + np.pi
        
        height = self.rho/(self.b*np.tan(self.Theta/180.*np.pi))
        if abs(height - 1.) < 0.01:
            height = 1.
        elif abs(height + 1.) < 0.01:
            height = -1.
        
        if self.below_equator and \
                local_v_limit > np.pi + np.arcsin(height):
            local_v_limit = np.pi  + np.arcsin(height)
        elif (not self.below_equator) and local_v_limit*180./np.pi > 180.:
            local_v_limit = np.pi
        
        return local_v_limit

    def self_obscuration_equations(self, p):
        # equation definitions for the self-obscuration solver
        
        x, y, z = p
        if self.geometry_code == 0:
            brack = self.R + self.a*np.cos(x)
            v_limit_t = - np.arctan(self.b*np.sin(z)*np.tan(self.inc_tot)/self.a) + np.pi
            brack2 = self.R + self.a*np.cos(v_limit_t)
            
            eq1 = brack*np.cos(TorusModel.u_m) - brack2*np.cos(z)
            eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                    - brack2*np.sin(z)
            eq3 = self.b*np.sin(x) + y*np.cos(self.inc_tot)\
                    - self.b*np.sin(v_limit_t)
        elif self.geometry_code == 2:
            brack = self.r_in + (self.rho-self.r_in)*np.tan(self.Theta/180.*np.pi)*abs(x)/self.rho
            eq1 = brack*np.cos(TorusModel.u_m) - self.rho*np.cos(z)
            eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                    - self.rho*np.sin(z)
            eq3 = x + y*np.cos(self.inc_tot)\
                    - self.rho/np.tan(self.Theta/180.*np.pi)
        else:
            brack = self.R + self.a*abs(np.cos(x))
            eq1 = brack*np.cos(TorusModel.u_m) - (self.R+self.a)*np.cos(z)
            eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                    - (self.R+self.a)*np.sin(z)
            eq3 = self.b*(1.-np.sin(x)) + y*np.cos(self.inc_tot) - self.b
        
        return [eq1, eq2, eq3]
    
    def self_obscuration_equations_special(self, p):
        # equation definitions for the self-obscuration solver
        
        x, y, z = p
        brack = self.R + self.a*abs(np.cos(x))
        eq1 = brack*np.cos(TorusModel.u_m) - (self.R+self.a)*np.cos(z)
        eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                - (self.R+self.a)*np.sin(z)
        eq3 = -self.b*(1.+np.sin(x)) + y*np.cos(self.inc_tot) - self.b
        
        return [eq1, eq2, eq3]
    
    def self_obscuration_equations_bottom(self, p):
        # equation definitions for the self-obscuration solver
        
        x, y, z = p
        if self.geometry_code == 2:
            brack = self.r_in + (self.rho-self.r_in)*np.tan(self.Theta/180.*np.pi)*abs(x)/self.rho
            eq1 = brack*np.cos(TorusModel.u_m) - self.r_in*np.cos(z)
            eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                    - self.r_in*np.sin(z)
            eq3 = x + y*np.cos(self.inc_tot)
        else:
            brack = self.R + self.a*abs(np.cos(x))
            eq1 = brack*np.cos(TorusModel.u_m) - self.R*np.cos(z)
            eq2 = brack*np.sin(TorusModel.u_m) + y*np.sin(self.inc_tot) \
                    - self.R*np.sin(z)
            eq3 = -self.b*(1.+np.sin(x)) + y*np.cos(self.inc_tot)
        
        return [eq1, eq2, eq3]
    
    def calculate_self_obscuration_line(self):
        # equation solver for the self-obscuration boundary
        
        if self.geometry_code == 0:
            if self.Theta < 15. and self.inc_tot > 0.866:
                res =  least_squares(self.self_obscuration_equations, (2, 10, 1), \
                    bounds = ((np.pi/2., 0, 0), (3.*np.pi/2., 10**10, np.pi)))
            else:
                res =  least_squares(self.self_obscuration_equations, (2, 1, 1), \
                        bounds = ((np.pi/2., 0, 0), (3.*np.pi/2., 10**10, np.pi)))
            x = res.x[0]  
            v_selfobs = x
            height = self.rho/(self.b*np.tan(self.Theta/180.*np.pi))
            if abs(height - 1.) < 0.01:
                height = 1.
            elif abs(height + 1.) < 0.01:
                height = -1.
            if self.below_equator and \
                    v_selfobs > np.pi + np.arcsin(height):
                v_selfobs = np.pi  + np.arcsin(height)
            elif (not self.below_equator) and v_selfobs*180./np.pi > 180.:
                v_selfobs = np.pi
            selfobs = v_selfobs
        elif self.geometry_code == 2:
            Zbrack = (self.r_in+abs(TorusModel.v_m)*(self.rho-self.r_in)*np.tan(self.Theta/180.*np.pi)/self.rho) 
            
            sinu = ((Zbrack/self.rho)**2.-1.+(np.tan(self.inc_tot)*(1./np.tan(self.Theta/180.*np.pi)-TorusModel.v_m/self.rho))**2.)*  \
                    self.rho/(2.*Zbrack*np.tan(self.inc_tot)*(TorusModel.v_m/self.rho-1./np.tan(self.Theta/180.*np.pi)))
            
            if -1. <= sinu <= 1.:
                uselfobs = (np.pi - np.arcsin(sinu))
            elif sinu < -1.:
                uselfobs = 2.*np.pi + 0.01
            else:
                uselfobs = -0.01
            
            selfobs = uselfobs
        
        else:
            # here we calculate u instead of v
            if TorusModel.v_m <= np.pi/2.:
                sinu = (1.-(self.R+self.a*abs(np.cos(TorusModel.v_m)))**2./(self.rho)**2. - \
                        (np.sin(TorusModel.v_m)*np.tan(self.inc_tot)/ \
                         np.tan(self.Theta/180.*np.pi))**2.)*np.tan(self.Theta/180.*np.pi)* \
                         self.rho/2./np.tan(self.inc_tot)/np.sin(TorusModel.v_m)/ \
                             (self.R+self.a*abs(np.cos(TorusModel.v_m)))
                if -1. <= sinu <= 1.:
                    uselfobs = (np.pi - np.arcsin(sinu))
                elif sinu < -1.:
                    uselfobs = 2.*np.pi + 0.01
                else:
                    uselfobs = -0.01
            else:
                sinu = (1.-(self.R+self.a*abs(np.cos(TorusModel.v_m)))**2./(self.rho)**2. - \
                        ((np.sin(TorusModel.v_m)+2.)*np.tan(self.inc_tot)/ \
                         np.tan(self.Theta/180.*np.pi))**2.)*np.tan(self.Theta/180.*np.pi)* \
                         self.rho/2./np.tan(self.inc_tot)/(np.sin(TorusModel.v_m)+2.)/ \
                             (self.R+self.a*abs(np.cos(TorusModel.v_m)))
                if -1. <= sinu <= 1.:
                    uselfobs = (np.pi - np.arcsin(sinu))
                elif sinu < -1.:
                    uselfobs = 2.*np.pi + 0.01
                else:
                    uselfobs = -0.01
            
            selfobs = uselfobs
        
        return selfobs
    
    def calculate_self_obscuration_line_bottom(self):
        # equation solver for the self-obscuration boundary
        
        if self.geometry_code == 2:
            if TorusModel.v_m < 0.:
                Zbrack = (self.r_in+abs(TorusModel.v_m)*(self.rho-self.r_in)*np.tan(self.Theta/180.*np.pi)/self.rho) 
                sinu = ((Zbrack/self.r_in)**2.-1.+(np.tan(self.inc_tot)*(TorusModel.v_m/self.r_in))**2.)*  \
                        self.r_in*self.r_in/(2.*Zbrack*np.tan(self.inc_tot)*(TorusModel.v_m))
                if -1. <= sinu <= 1.:
                    u_selfobs_bottom = (np.pi - np.arcsin(sinu))
                else:
                    u_selfobs_bottom = -0.01
            selfobs_bottom = u_selfobs_bottom
        
        else:
            if TorusModel.v_m >= np.pi:
                sinu = (1.-(self.R+self.a*abs(np.cos(TorusModel.v_m)))**2./(self.R)**2. - \
                        ((np.sin(TorusModel.v_m)+1.)*np.tan(self.inc_tot)*self.rho/self.R/ \
                         np.tan(self.Theta/180.*np.pi))**2.)*np.tan(self.Theta/180.*np.pi) \
                         /self.rho/2./np.tan(self.inc_tot)/ \
                        (np.sin(TorusModel.v_m)+1.)/(self.R+self.a*abs(np.cos(TorusModel.v_m)))
                if -1. <= sinu <= 1.:
                    u_selfobs_bottom = (np.pi - np.arcsin(sinu))
                else:
                    u_selfobs_bottom = -0.01
            selfobs_bottom = u_selfobs_bottom
        
        return selfobs_bottom

    def visible_any(self):
        # loads pre-computed visibility condition from table and provides
        # True or False, if visible per given Theta and mue
        
        visible = True
        if self.geometry_code == 0:
            # Define the equation to solve
            def equation(y):
                vlim = - np.arctan(y) + np.pi
                return (- fR/fa - (frho)/fa) - np.cos(vlim) + y*(np.sin(vlim) - fhval/fb)
            
            # Define a range of x values
            x_values = np.linspace(0./180.*np.pi+0.00000001, 90./180.*np.pi-0.000000001, 40000)

            # Solve for y values
            y_values = []
            for x in x_values:
                y_guess = 100000.
                frin = self.r_in
                frho = self.rho
                fhval = frho / np.tan(x)
                fa = frin*(frho-frin)/(2.*frin-frho)
                fb = (-fhval*fa*fa/(frho-fa-frin)/ np.tan(x))**(0.5)
                fR = fa + frin                
                fhval = frho/np.tan(x)
                y_solution = fsolve(equation, y_guess, xtol=1.49012e-10, maxfev=500000)
                inclim = np.arctan(y_solution[0]*fa/fb)
                
                # Limit y_solution to the range [-π/2, π/2] before appending
                if -np.pi/2 <= inclim <= np.pi/2:
                    y_values.append(inclim)
                else:
                    y_values.append(np.nan) # Append NaN for values outside the range

            # Corresponding x values for the valid (non-NaN) y values
            x_values_filtered = x_values[~np.isnan(y_values)]

            # Filter out NaN values from y_values for plotting
            y_values_filtered = [y for y in y_values if not np.isnan(y)]
            
            # order reverted
            interp_func_x = interp1d(x_values_filtered, y_values_filtered, kind='linear', \
                                     fill_value="extrapolate")
            inc_limit = interp_func_x(self.Theta/180.*np.pi)
            
            if self.inc_tot > inc_limit:
                visible = False
        
        return visible

    def surface_info_bc(self):
        # get information in the {u,v} grid that does not need to be computed 
        # per each primary polarization state, and return which local points
        # are visible and what is their area contribution
        
        # in default all u's and v's are omitted, we change the values to 
        # "False", if subset conditions on the surface are fulfilled
        flag_omit = np.full((self.N_u, self.N_v), True)
        areas = np.zeros((self.N_u, self.N_v))
        
        for v, v_val in enumerate(self.v_grid):
            TorusModel.v_m = v_val[0]
            if self.geometry_code == 2:
                if self.below_equator and v_val[0] < 0.:
                    # the equatorial circle can cover some parts of the
                    # walls in the bottom half too
                    selfobs_bottom = self.calculate_self_obscuration_line_bottom()
            else:
                if self.below_equator and v_val[0] >= np.pi:
                    # the equatorial circle can cover some parts of the
                    # walls in the bottom half too
                    selfobs_bottom = self.calculate_self_obscuration_line_bottom()
            selfobs = self.calculate_self_obscuration_line()
            
            for u, u_val in enumerate(self.u_grid): 
                # line-of-sight conditions for center point (with high 
                # enough resolution this will not matter) in v
                lp = LocalPoint(u_val[0], v_val[0])
                
                # let's test that we don't have any bins that are not 
                # visible due to full opacity
                if not (0. < np.arccos(LocalPoint(u_val[0], \
                        v_val[1]).loc_mue) < np.pi/2. and 0. < \
                        np.arccos(lp.loc_mui) < np.pi/2.):
                    # if mui condition fails, it shouldn't, increase N_v
                    continue
                if self.geometry_code == 2:
                    if u_val[0] < selfobs:
                        continue
                    if self.below_equator and v_val[0] < 0. and u_val[0] < selfobs_bottom:
                        continue
                else:
                    if u_val[0] < selfobs and (0. <= v_val[0] <= np.pi/2.) and selfobs >= 0.:
                        continue
                    if u_val[0] < selfobs and np.pi <= v_val[0] <= 3.*np.pi/2. and selfobs >= np.pi:
                        continue
                    if u_val[0] > selfobs and np.pi <= v_val[0] <= 3.*np.pi/2. and selfobs < np.pi:
                        continue
                    if self.below_equator and v_val[0] >= np.pi and not (u_val[0] > selfobs_bottom and selfobs_bottom > 0.):
                        continue
                    
                # mark the region
                flag_omit[u][v] = False
                
                # compute the local area contribution
                areas[u][v] = lp.surface(u_val[1], u_val[2], v_val[1], \
                                         v_val[2])     

        return flag_omit, areas


    def surface_info_t(self):
        # get information in the {u,v} grid that does not need to be computed 
        # per each primary polarization state, and return which local points
        # are visible and what is their area contribution
        
        # in default all u's and v's are omitted, we change the values to 
        # "False", if subset conditions on the surface are fulfilled
        flag_omit = np.full((self.N_u, self.N_v), True)
        areas = np.zeros((self.N_u, self.N_v))
        
        for u, u_val in enumerate(self.u_grid):         
            # get the limitting curves
            TorusModel.u_m = u_val[0]
            local_v_limit = self.calculate_v_lim()
            selfobs = self.calculate_self_obscuration_line()
            
            for v, v_val in enumerate(self.v_grid):
                # line-of-sight conditions for center point (with high 
                # enough resolution this will not matter) in v
                lp = LocalPoint(u_val[0], v_val[0])

                # let's test that we don't have any bins that are not 
                # visible due to full opacity
                if not (0. < np.arccos(LocalPoint(u_val[0], \
                        v_val[1]).loc_mue) < np.pi/2. and 0. < \
                        np.arccos(lp.loc_mui) < np.pi/2.):
                    # if mui condition fails, it shouldn't, increase N_v
                    continue
                if v_val[0] > local_v_limit:
                    # invisible from this inclination
                    continue
                if v_val[0] > selfobs:
                    continue
                
                # mark the region
                flag_omit[u][v] = False
                
                # compute the local area contribution
                areas[u][v] = lp.surface(u_val[1], u_val[2], v_val[1], \
                                         v_val[2])                    

        return flag_omit, areas
    
    def get_2D_coords(self, U, V):
        # Compute torus cartesian coordinates in 3D space, rotate and project
    
        if self.geometry_code == 0:
            X = (self.R + self.a * np.cos(V)) * np.cos(U)
            Y = (self.R + self.a * np.cos(V)) * np.sin(U)
            Z = self.b * np.sin(V)
        elif self.geometry_code == 1:
            X = (self.R + self.a * abs(np.cos(V))) * np.cos(U)
            Y = (self.R + self.a * abs(np.cos(V))) * np.sin(U)
            if V <= np.pi/2.:
                Z = self.b * (1. - np.sin(V))
            else:
                Z = - self.b * (1. + np.sin(V))
        else:
            brack = self.r_in + (self.rho-self.r_in)*np.tan(self.Theta/180.*np.pi)*abs(V)/self.rho
            X = brack * np.cos(U)
            Y = brack * np.sin(U)
            Z = V
        
        # Apply rotation for viewing angle around X axis
        Y_rot = -Y * np.sin(np.pi/2. - self.inc_tot) + \
                    Z * np.cos(np.pi/2. - self.inc_tot)
        X_rot = X
        
        # Project torus onto 2D plane (orthographic projection)
        X2D, Y2D = X_rot, Y_rot
    
        return X2D, Y2D
    
    def scale_2D_coords(self, X2D, Y2D, xmin, xmax, ymin, ymax):
        # scales the projection, so that the X and Y axis have correct ranges
        
        if ymax <= xmax:
            scaled_X2D = (X2D - xmin) / (xmax - xmin) * (self.image_resolution )
            scaled_Y2D = (Y2D - ymin) / (ymax - ymin) * \
                        ((self.image_resolution )  * ymax / xmax) - \
                        (self.image_resolution ) * ymax / xmax / 2. + \
                            (self.image_resolution ) / 2.
        else:
            scaled_X2D = (X2D - xmin) / (xmax - xmin) * (self.image_resolution ) \
                * xmax / ymax - (self.image_resolution ) * xmax / ymax / 2. + \
                (self.image_resolution ) / 2.
            scaled_Y2D = (Y2D - ymin) / (ymax - ymin) * \
                        ((self.image_resolution) )
        
        return scaled_X2D, scaled_Y2D
    
    def initiate_coords(self):
        # initiates arrays important for image production
        
        # for the grid for drawing a background torus
        U_arr = np.asarray(np.linspace(0, 2*np.pi, 31))
        if self.geometry_code == 0:
            V_arr = np.asarray(np.linspace(0, 2*np.pi, 40))
        elif self.geometry_code == 2:
            V_arr = np.asarray(np.linspace(-self.rho/(np.tan(self.Theta/180.*np.pi)), \
                                           self.rho/(np.tan(self.Theta/180.*np.pi)), 40))
        else:
            V_arr = np.asarray(list(np.linspace(0, np.pi/2., 20))+ \
                               list(np.linspace(np.pi, 3.*np.pi/2., 20)))
            
        X2Ds = []
        Y2Ds = []
        for u in range(len(U_arr)-1):
            for v in range(len(V_arr)-1):
                X2D_1, Y2D_1 = self.get_2D_coords(U_arr[u], V_arr[v])
                X2Ds.append(X2D_1)
                Y2Ds.append(Y2D_1)
        X2Ds, Y2Ds = np.asarray(X2Ds), np.asarray(Y2Ds)
        
        if self.geometry_code == 1:
            verts1 = []
            verts2 = []
            for u in range(len(U_arr)-1):
                for v in range(int(len(V_arr)/2.-1)):
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v])
                    X2D_a, Y2D_a = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v+1])
                    X2D_b, Y2D_b = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v+1])
                    X2D_c, Y2D_c = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v])
                    X2D_d, Y2D_d = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    vert = [
                                    (X2D_a, -Y2D_a+self.image_resolution),
                                    (X2D_b, -Y2D_b+self.image_resolution),
                                    (X2D_c, -Y2D_c+self.image_resolution),
                                    (X2D_d, -Y2D_d+self.image_resolution)
                    ]
                    verts1.append(vert)
                    
                for v in range(int(len(V_arr)/2.-1)):
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v+int(len(V_arr)/2.)])
                    X2D_a, Y2D_a = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v+1+int(len(V_arr)/2.)])
                    X2D_b, Y2D_b = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v+1+int(len(V_arr)/2.)])
                    X2D_c, Y2D_c = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v+int(len(V_arr)/2.)])
                    X2D_d, Y2D_d = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    vert = [
                                    (X2D_a, -Y2D_a+self.image_resolution),
                                    (X2D_b, -Y2D_b+self.image_resolution),
                                    (X2D_c, -Y2D_c+self.image_resolution),
                                    (X2D_d, -Y2D_d+self.image_resolution)
                    ]
                    verts2.append(vert)
            
            verts = [verts1,verts2]  
        else:
            verts = []
            for u in range(len(U_arr)-1):
                for v in range(len(V_arr)-1):
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v])
                    X2D_a, Y2D_a = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u], V_arr[v+1])
                    X2D_b, Y2D_b = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v+1])
                    X2D_c, Y2D_c = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    tmp_X2D, tmp_Y2D = self.get_2D_coords(U_arr[u+1], V_arr[v])
                    X2D_d, Y2D_d = self.scale_2D_coords(tmp_X2D, tmp_Y2D, \
                                                        X2Ds.min(), X2Ds.max(), \
                                                        Y2Ds.min(), Y2Ds.max())
                    vert = [
                                    (X2D_a, -Y2D_a+self.image_resolution),
                                    (X2D_b, -Y2D_b+self.image_resolution),
                                    (X2D_c, -Y2D_c+self.image_resolution),
                                    (X2D_d, -Y2D_d+self.image_resolution)
                    ]
                    verts.append(vert)
                
        # initiate the X and Y grids to save Stokes parameters in Sxy there
        if Y2Ds.max() <= X2Ds.max():
            grid_nonscaled = np.asarray(np.linspace(X2Ds.min(), X2Ds.max(), \
                                                    self.image_resolution+1))
            gridx = (grid_nonscaled - X2Ds.min()) / (X2Ds.max() - X2Ds.min()) * \
                        (self.image_resolution)
            gridy = (grid_nonscaled - Y2Ds.min()) / (Y2Ds.max() - Y2Ds.min()) * \
                        (self.image_resolution)
        else:
            grid_nonscaled = np.asarray(np.linspace(Y2Ds.min(), Y2Ds.max(), \
                                                    self.image_resolution+1))
            gridx = (grid_nonscaled - X2Ds.min()) / (X2Ds.max() - X2Ds.min()) * \
                        (self.image_resolution )
            gridy = (grid_nonscaled - Y2Ds.min()) / (Y2Ds.max() - Y2Ds.min()) * \
                        (self.image_resolution)
        
        deltax = abs(gridx[1]-gridx[0])/2.
        deltay = abs(gridy[1]-gridy[0])/2.
        
        Sxy = np.zeros((3,len(gridx),len(gridx)))
        IQUxy = []
        for er in range(len(self.image_energy_ranges)):
            IQUxy.append(Sxy)

        return X2Ds, Y2Ds, gridx, gridy, deltax, deltay, IQUxy, verts
    
    
    def calculate_area_fraction(self, X2Ds, Y2Ds, x0, y0, deltax, deltay, u1, v1, u2, \
                                v2):
        # returns the fraction of area of local polygon in U and V that
        # intersects in projection with the pixel in X and Y for imaging
        
        x_low = x0
        x_high = x0+2.*deltax
        y_low = y0
        y_high = y0+2.*deltay
        
        tmp_X2D, tmp_Y2D = self.get_2D_coords(u1, v1)
        X2D_a, Y2D_a = self.scale_2D_coords(tmp_X2D, tmp_Y2D, X2Ds.min(), \
                                            X2Ds.max(), Y2Ds.min(), Y2Ds.max())
        tmp_X2D, tmp_Y2D = self.get_2D_coords(u1, v2)
        X2D_b, Y2D_b = self.scale_2D_coords(tmp_X2D, tmp_Y2D, X2Ds.min(), \
                                            X2Ds.max(), Y2Ds.min(), Y2Ds.max())
        tmp_X2D, tmp_Y2D = self.get_2D_coords(u2, v2)
        X2D_c, Y2D_c = self.scale_2D_coords(tmp_X2D, tmp_Y2D, X2Ds.min(), \
                                            X2Ds.max(), Y2Ds.min(), Y2Ds.max())
        tmp_X2D, tmp_Y2D = self.get_2D_coords(u2, v1)
        X2D_d, Y2D_d = self.scale_2D_coords(tmp_X2D, tmp_Y2D, X2Ds.min(), \
                                            X2Ds.max(), Y2Ds.min(), Y2Ds.max())

        polygon_a_coords = [(X2D_a, Y2D_a), (X2D_b, Y2D_b), (X2D_c, Y2D_c), \
                            (X2D_d, Y2D_d)]
        polygon_b_coords = [(x_low,y_low), (x_high, y_low),(x_high, y_high), \
                            (x_low, y_high)]
        
        # Create Polygon objects from the given coordinates
        polygon_a = Polygon(polygon_a_coords)
        polygon_b = Polygon(polygon_b_coords)
        
        # Calculate the intersection area
        intersection_area = polygon_a.intersection(polygon_b).area
        
        # Calculate the total area of polygon A
        polygon_a_area = polygon_a.area
        
        # Calculate and return the area fraction
        area_fraction = intersection_area / \
                        polygon_a_area if polygon_a_area > 0. else 0.
        
        return area_fraction
                                            
    def spectra_energy_integration(self, IQU_loc_spectrum, er_vals):
        # integrates the spectra stored in [cts cm^-2 s^-1] at a selected 
        # energy range
        
        B_lower = er_vals[0]
        B_upper = er_vals[1]
        IQU_integrated = 0.
        for e in range(len(IQU_loc_spectrum)):
            A_lower = self.energies[0][e]
            A_upper = self.energies[1][e]
            A_mid = (A_lower+A_upper)/2.
            if not (A_upper < B_lower or B_upper < A_lower):
                intersection_lower = max(A_lower, B_lower)
                intersection_upper = min(A_upper, B_upper)
                
                # Calculate the length of the intersection
                intersection_length = intersection_upper - intersection_lower
                
                # Calculate the length of interval A
                A_length = A_upper - A_lower
                
                # Calculate the fraction of A that intersects with B and store in [keV cm^-2 s^-1]
                fraction = intersection_length / A_length
                IQU_integrated = IQU_integrated + IQU_loc_spectrum[e]*fraction*A_mid

        return IQU_integrated
    
    @staticmethod
    def custom_formatter(value, pos):
        # A custom formatter function for plotting
        
        if value == 0:
            return ""  # Omit 0
        else:
            return f"{int(round(value))}"
    
    def create_image(self, gridx, gridy, deltax, deltay, IQUxy_e, image_name, verts, er_vals):
        # adjusts the projected data, plots and saves image data
        
        # adjust the collected I, Q, U in X and Y coordinates
        if self.image_resolution % 2 == 1:
            # is odd
            max_halfspace = (self.image_resolution+1)/2.
        else:
            # is even
            max_halfspace = (self.image_resolution)/2.
        
        for x in range(int(max_halfspace)):
            for y in range(len(gridy)-1):
                # I and Q are added symmetrically, U is added with minus sign
                if self.image_resolution % 2 == 1:
                    if x == max_halfspace - 1:
                        IQUxy_e[0][x][y] = 2.*IQUxy_e[0][-x-2][y]
                        IQUxy_e[1][x][y] = 2.*IQUxy_e[1][-x-2][y]
                        IQUxy_e[2][x][y] = 0.
                    else:
                        IQUxy_e[0][x][y] = IQUxy_e[0][-x-2][y]
                        IQUxy_e[1][x][y] = IQUxy_e[1][-x-2][y]
                        IQUxy_e[2][x][y] = -IQUxy_e[2][-x-2][y]
                else:
                    IQUxy_e[0][x][y] = IQUxy_e[0][-x-2][y]
                    IQUxy_e[1][x][y] = IQUxy_e[1][-x-2][y]
                    IQUxy_e[2][x][y] = -IQUxy_e[2][-x-2][y]
        
        # store PD, PA, pF/F*
        PDs = np.zeros((len(gridx)-1,len(gridx)-1))
        PAs = np.zeros((len(gridx)-1,len(gridx)-1))
        PFs = np.zeros((len(gridx)-1,len(gridx)-1))
        x_startpoint = []
        x_endpoint = []
        y_startpoint = []
        y_endpoint = []
        for x in range(len(gridx)-1):
            for y in range(len(gridy)-1):
                # density of local tables in cm-3 for normalization
                n_H = 10.**15. 
                # erg to keV factor
                ergkeV = 624150647.99632
                # energy in the middle of the imaging bin in keV
                #mid_energy = (er_vals[0] + er_vals[1])/2.
                # normalization K that was chosen in reflection model loading routine
                K_factor = 10.**(-14.)
                # correction for original power-law cutoffs and the total flux
                E_min = 10.**(-1.1)
                E_max = 10.**(2.4)
                if float(self.Gamma) == 2.:
                    scale_factor = np.log(er_vals[1]/er_vals[0])/np.log(E_max/E_min)
                else:
                    scale_factor = (er_vals[1]**(2.-float(self.Gamma))- \
                                    er_vals[0]**(2.-float(self.Gamma)))/ \
                                    (E_max**(2.-float(self.Gamma))- \
                                     E_min**(2.-float(self.Gamma)))
                
                a_factor = TorusModel.compute_a_mu(float(self.mue),self.inctype_code)
                if self.xi0_val != 0.:
                    F_star = n_H*abs(self.xi0_val)*ergkeV*scale_factor*a_factor/4./np.pi
                else:
                    print('error: currently it is not possible to plot images for this ionization type')
                    sys.exit()
                tmp = (IQUxy_e[1][x][y]*IQUxy_e[1][x][y] + \
                       IQUxy_e[2][x][y]*IQUxy_e[2][x][y])**.5
                if IQUxy_e[0][x][y] != 0.:
                    PDs[x][y] = tmp/IQUxy_e[0][x][y]
                else:
                    PDs[x][y] = 0.
                PAs[x][y] = 0.5*(np.arctan2(IQUxy_e[2][x][y],IQUxy_e[1][x][y]))
                PFs[x][y] = tmp/F_star/K_factor
        
                angle_deviation = PAs[x][y]
                line_length = 0.3*PDs[x][y]
                x_endpoint.append(gridx[x]+deltax + \
                                  np.cos(angle_deviation-np.pi/2.) * \
                                  line_length)
                y_endpoint.append(gridy[y]+deltay + \
                                  np.sin(angle_deviation-np.pi/2.) * \
                                  line_length)
                x_startpoint.append(gridx[x]+deltax - \
                                    np.cos(angle_deviation-np.pi/2.) * \
                                    line_length)
                y_startpoint.append(gridy[y]+deltay - \
                                    np.sin(angle_deviation-np.pi/2.) * \
                                    line_length)
                
        # plot and save the image in pdf
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # plotting the background torus
        if self.geometry_code == 1:
            # does not have continuous grid in v
            poly = PolyCollection(verts[0], facecolors=None, edgecolors='green', \
                              linewidths=1, alpha = 0.1)
            ax.add_collection(poly)
            poly = PolyCollection(verts[1], facecolors=None, edgecolors='green', \
                              linewidths=1, alpha = 0.1)
            ax.add_collection(poly)
        else:
            poly = PolyCollection(verts, facecolors=None, edgecolors='green', \
                              linewidths=1, alpha = 0.1)
            ax.add_collection(poly)
        
        # plotting the heatmap
        font = {'size'   : 22}
        matplotlib.rc('font', **font)
        y, x = np.meshgrid(gridy, gridx)
        z = PFs
        z_min, z_max = self.image_limits[0], self.image_limits[1]    
        c = ax.pcolormesh(x, y, z, cmap='OrRd', \
                norm=matplotlib.colors.LogNorm(vmin=z_min, vmax=z_max))
        ax.set_xticks(np.arange(0., float(self.image_resolution), 1), \
                      minor=True)
        ax.set_yticks(np.arange(0., float(self.image_resolution), 1), \
                      minor=True)
        formatter = FuncFormatter(TorusModel.custom_formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.axis([0.,self.image_resolution,0.,self.image_resolution])        
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(length=5, width=1, direction = 'in', pad = 2, \
                            which = 'major')
        cbar.ax.tick_params(length=2, width=1, direction = 'in', pad = 2, \
                            which = 'minor')
        cbar.ax.set_ylabel('$\mathit{p\,F\,/\,F_*}$', fontsize=15, labelpad=5)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(12)
        ax.tick_params('both', bottom = True, top = True, left = True, \
                       right = True, length=10, width=1, direction = 'in', \
                       pad = 2, which='major', labelsize=12)
        ax.tick_params('both', bottom = True, top = True, left = True, \
                       right = True, length=5, width=1, direction = 'in', \
                       which='minor', labelsize=12)
        [x.set_linewidth(1) for x in ax.spines.values()]        
        ax.set_ylabel('X', fontsize=15, labelpad=5)                                    
        ax.set_xlabel('Y', fontsize=15, labelpad=5)
        
        # plotting the PD and PA bars
        for x in range(len(x_startpoint)):
            ax.plot([x_startpoint[x], x_endpoint[x]], \
                    [y_startpoint[x], y_endpoint[x]], \
                    color='black', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.saving_directory + '/images/' + image_name +'.png', \
                    format='png')
        plt.close('all')
        
        # save image data into text file
        headers = ['X', 'Y', 'I', 'Q', 'U','PF', 'PD', 'PA']
        data = []
        for i, x in enumerate(gridx):
            for j, y in enumerate(gridy):
                if i == len(gridx)-1 or j == len(gridy)-1:
                    continue
                row = [round(x,3), round(y,3), IQUxy_e[0][i][j], \
                       IQUxy_e[1][i][j], IQUxy_e[2][i][j], PFs[i][j], \
                       PDs[i][j], PAs[i][j]]
                data.append(row)
        
        max_lengths = [max(len(str(item)) for item in column) for column \
                       in zip(*([headers] + data))]
        
        with open(self.saving_directory + '/images_data/' + image_name + \
                  '.dat', "w") as file:
            header_line = "\t".join(header.ljust(max_lengths[i]) for \
                                    i, header in enumerate(headers))
            file.write(header_line + "\n")            
            for row in data:
                row_line = "\t".join(str(item).ljust(max_lengths[i]) for \
                                     i, item in enumerate(row))
                file.write(row_line + "\n")
    
    def get_primary_spectrum(self):
        # compute the primary spectrum
        
        local_primary = []
        for e in range(len(self.energies[0])):
            delta_energy = abs(self.energies[1][e]-self.energies[0][e])
            n_H = 10.**15. 
            ergkeV = 624150647.99632
            K_factor = 10.**(-14.)
            E_min = 10.**(-1.1)
            E_max = 10.**(2.4)
            if float(self.Gamma) == 2.:
                scale_factor = np.log(self.energies[1][e]/self.energies[0][e])/np.log(E_max/E_min)
            else:
                scale_factor = (self.energies[1][e]**(2.-float(self.Gamma))- \
                                self.energies[0][e]**(2.-float(self.Gamma)))/ \
                                (E_max**(2.-float(self.Gamma))- \
                                 E_min**(2.-float(self.Gamma)))
            a_factor = TorusModel.compute_a_mu(float(self.mue),self.inctype_code)
            if self.xi0_val != 0.:
                F_star = n_H*abs(self.xi0_val)*ergkeV*scale_factor*a_factor/4./np.pi/delta_energy
            local_primary.append(F_star*K_factor)
        
        return local_primary
    
    def generator(self):
        # walk through the parameter space and compute the result per each        
        
        TorusModel.R_t = self.R
        TorusModel.a_t = self.a
        TorusModel.b_t = self.b
        TorusModel.r_in_t = self.r_in
        TorusModel.xi0_t = self.xi0
        TorusModel.xi0_val_t = self.xi0_val
        TorusModel.beta_t = self.beta
        TorusModel.rho_t = self.rho
        TorusModel.IQUs_t = self.IQUs   
        TorusModel.theta_t = self.Theta
        TorusModel.inc_t = self.inc_tot
        TorusModel.geometry_code_t = self.geometry_code
        TorusModel.inctype_code_t = self.inctype_code
        TorusModel.Gamma_t = self.Gamma
        
        # get information in the {u,v} grid that does not need to be computed
        # per each primary polarization state
        if float(self.Theta_input) > 1.:
            flag_visibility = self.visible_any()
        else:
            flag_visibility = True
        if self.geometry_code == 0:
            flag_omit, areas = self.surface_info_t()
        else:
            flag_omit, areas = self.surface_info_bc()

        # load all table data and create images, if required
        for pp in self.primpols:
            if os.path.isfile(self.name_file(pp) +'.dat'):
                continue

            # define
            name = self.name_file(pp)                    
            final_spectra = []
            final_spectra_prim = []
            for iqu, name_IQU in enumerate(self.IQUs):
                final_spectra.append([0.]*len(self.energies[0]))
                final_spectra_prim.append([0.]*len(self.energies[0]))

            if (pp[0],self.mue,self.Theta_input,self.Gamma, self.r_in_input, \
                    self.rho_input, self.xi0_input, self.beta_input) not in self.image_list:
                # calculate empty X and Y grid
                local_X2Ds, local_Y2Ds, gridx, gridy, deltax, deltay, IQUxy, \
                        verts = self.initiate_coords()
            
            if self.produce_primary == True and self.inc_tot < self.Theta/180.*np.pi:
                for f in range(len(final_spectra_prim)):
                    if f == 0:
                        final_spectra_prim[f] = list(map(add, \
                                                final_spectra_prim[f], \
                                                self.get_primary_spectrum()))
                    if f == 1:
                        if self.inctype_code != 2 and self.inctype_code != 3:
                            p0 = pp[1]
                            Psi0 = pp[2]
                        elif self.inctype_code == 2:
                            p0 = 0.064*(1.-np.cos(self.inc_tot))*(1.+16.3*np.cos(self.inc_tot)+ \
                                        6.2*np.cos(self.inc_tot)* \
                                        np.cos(self.inc_tot))/(1.+8.2*np.cos(self.inc_tot)- \
                                        2.1*np.cos(self.inc_tot)*np.cos(self.inc_tot))
                            Psi0 = 0.
                        elif self.inctype_code == 3:
                            p0 = 0.117*(1.-np.cos(self.inc_tot))/(1.+3.582*np.cos(self.inc_tot))
                            Psi0 = np.pi/2.
                        
                        Q_vals = map(lambda xxx: \
                                        xxx*p0*np.cos(2.*Psi0), \
                                        final_spectra_prim[0])
                        final_spectra_prim[f] = list(map(add, \
                                                final_spectra_prim[f], \
                                                Q_vals))
                    if f == 2:
                        U_vals = map(lambda xxx: \
                                        xxx*p0*np.sin(2.*Psi0), \
                                        final_spectra_prim[0])
                        final_spectra_prim[f] = list(map(add, \
                                                final_spectra_prim[f], \
                                                U_vals))

            # check, if inclination not over visibility limit by Theta
            if flag_visibility:
                for u, u_val in enumerate(self.u_grid):                    
                    for v, v_val in enumerate(self.v_grid):
                        # we see this area and take it into account
                        if flag_omit[u][v] == False:
                            lp = LocalPoint(u_val[0], v_val[0])
                            
                            # interpolate and rotate
                            IQU_loc = lp.interpolate(self.parameters, \
                                                     self.all_spectra, \
                                                     self.all_spectra_neutral, \
                                                     self.energies, \
                                                     pp[1], pp[2])
                            # comment out the following line, if interested 
                            # only in intensity and faster computation
                            IQU_loc_final = lp.rotate(IQU_loc)
                            
                            # iterate in I, Q, U
                            for f in range(len(final_spectra)):
                                # if imaging requested
                                if (pp[0],self.mue,self.Theta_input, \
                                        self.Gamma, self.r_in_input, \
                                        self.rho_input, self.xi0_input, \
                                        self.beta_input) not in self.image_list:
                                    # add values in X and Y pixels
                                    for x in range(len(gridx)-1):
                                        if x < (len(gridx)-1)/2.-2:
                                            # first half-space in u is not 
                                            # computed anyways, we later 
                                            # symmetrically add the results
                                            continue
                                        
                                        for y in range(len(gridy)-1):
                                            # populate the second half-space
                                            area_fraction = \
                                                self.calculate_area_fraction( \
                                                      local_X2Ds, local_Y2Ds, \
                                                      gridx[x], gridy[y], deltax, deltay,\
                                                      -u_val[1]+np.pi, \
                                                      v_val[1], \
                                                      -u_val[2]+np.pi, \
                                                      v_val[2])
                                            if area_fraction > 0.:
                                                for er, er_vals in enumerate( \
                                                     self.image_energy_ranges):
                                                    IQU_loc_integrated = \
                                              self.spectra_energy_integration(\
                                              IQU_loc_final[f], er_vals)
                                                    IQUxy[er][f][x][y] = \
                                              IQUxy[er][f][x][y] + \
                                              IQU_loc_integrated * \
                                              area_fraction * areas[u][v]
                                            
                                if f != 2:
                                    # U's are all zero, so we do not 
                                    # include f = 2; add and multiply by 2
                                    # to add the second half-space in u
                                    integrand = map(lambda x: \
                                                    x*areas[u][v]*2., \
                                                    IQU_loc_final[f])
                                    
                                    final_spectra[f] = list(map(add, \
                                                            final_spectra[f], \
                                                            integrand))
                                
            # create image, if requested
            if (pp[0],self.mue,self.Theta_input,self.Gamma, self.r_in_input, \
                self.rho_input, self.xi0_input, self.beta_input) not in self.image_list:
                for er, er_vals in enumerate(self.image_energy_ranges):
                    image_name = self.name_image(er_vals, pp)
                    self.create_image(gridx, gridy, deltax, deltay, IQUxy[er], image_name, \
                                      verts, er_vals)
                    
            yield (name, self.energies[0], self.energies[1], final_spectra, final_spectra_prim)
        
    def save_ascii(self, name, ener_lo, ener_hi, final_spectra):
        # save one ASCII file named accroding to other parameters and 
        # containing low & high energy bin boundaries and the spectra I, Q, U
        
        final_names = ['ENERGY_LO','ENERGY_HI']
        for iqu in self.IQUs:
            final_names += iqu

        final_data = [ener_lo, ener_hi] + final_spectra
        ascii.write(final_data, name+'.dat', names =  final_names, \
                                                        overwrite=True)
        
        '''
        # alternatively, if the ascii library is not working
        
        check_dirs = open(name+'.dat','w+')  
        writeline = 'ENERGY_LO'+'\t'+'ENERGY_HI'+'\t'+'I'+'\t'+'Q'+'\t'+'U'
        check_dirs.write(writeline+"\n")
        for i in range(len(ener_lo)):
            writeline = str(ener_lo[i])+'\t'+str(ener_hi[i])
            for iqu_final in final_spectra:
                writeline += '\t'+str(iqu_final[i])
            check_dirs.write(writeline+"\n")
        check_dirs.close()  
        '''
        
    def save_ascii_prim(self, name, ener_lo, ener_hi, final_spectra_prim):
        final_names = ['ENERGY_LO','ENERGY_HI']
        for iqu in self.IQUs:
            final_names += iqu

        final_data = [ener_lo, ener_hi] + final_spectra_prim
        ascii.write(final_data, name+'_prad.dat', names =  final_names, \
                                                        overwrite=True)

class LocalPoint:

    def __init__(self, u_point, v_point):
        """
        Make the angular computations and table interpolations required at 
        each illuminated point of the torus surface.
        
        Args:
            u_point: a float of u in radians defining a point on the torus 
                        surface, typically u_mid of one bin
            v_point: a float of v in radians defining a point on the torus 
                        surface, typically v_mid of one bin            
        """
        
        self.u_point = u_point
        self.v_point = v_point
        
        self.x, self.y, self.z = self.get_xyz()
        
        self.distance = (self.x * self.x + self.y * self.y + self.z * \
                         self.z )**.5
        self.dPdx, self.dPdy, self.dPdz = \
                    self.compute_derivatives()
        
        self.E = [0.,np.sin(TorusModel.inc_t), \
                  np.cos(TorusModel.inc_t)] # normalized
        self.I = [self.x, self.y, self.z] # not normalized   
        self.z_vec = [0., 0., 1.]
        self.n_norm, self.n = self.compute_norm_vector()
        
        if self.z >= 0.:
            self.dU = [-np.sin(self.u_point), np.cos(self.u_point), 0.]
        else:
            self.dU = [np.sin(self.u_point), -np.cos(self.u_point), 0.]
        
        self.In = self.I[0]*self.n[0] + self.I[1]*self.n[1] + \
                    self.I[2]*self.n[2]
        self.En = self.E[0]*self.n[0] + self.E[1]*self.n[1] + \
                    self.E[2]*self.n[2]
        self.Ez = self.E[0]*self.z_vec[0] + self.E[1]*self.z_vec[1] + \
                    self.E[2]*self.z_vec[2]
        self.Ip, self.Ep = self.projected_vectors()
        self.n_p, self.zp, self.y_vec = self.compute_E_projected() # normalized

        self.loc_mui = self.compute_loc_mui()
        self.loc_mue = self.compute_loc_mue()
        self.loc_Phi = self.compute_loc_Phi()     
        self.cosscat_ang = self.compute_cosscat_ang()
        self.loc_iang = self.compute_loc_iang()
        self.a_mu = TorusModel.compute_a_mu(abs(self.loc_iang),TorusModel.inctype_code_t)
        self.loc_xi = self.compute_loc_xi()
        
    def get_xyz(self):
        # compute the local values of Cartesian x, y, z
        
        if TorusModel.geometry_code_t == 0:
            x = (TorusModel.R_t + TorusModel.a_t*np.cos(self.v_point)) * \
                        np.cos(self.u_point)
            y = (TorusModel.R_t + TorusModel.a_t*np.cos(self.v_point)) * \
                        np.sin(self.u_point)
            z = TorusModel.b_t*np.sin(self.v_point)
        elif TorusModel.geometry_code_t == 1:
            x = (TorusModel.R_t + TorusModel.a_t*abs(np.cos(self.v_point))) * \
                        np.cos(self.u_point)
            y = (TorusModel.R_t + TorusModel.a_t*abs(np.cos(self.v_point))) * \
                        np.sin(self.u_point)
            if self.v_point <= np.pi/2.:
                z = TorusModel.b_t*(1.-np.sin(self.v_point))
            else:
                z = -TorusModel.b_t*(1.+np.sin(self.v_point))
        else:
            brack = TorusModel.r_in_t + (TorusModel.rho_t-TorusModel.r_in_t) \
                        * np.tan(TorusModel.theta_t/180.*np.pi) * abs(self.v_point) / TorusModel.rho_t
            x = brack * np.cos(self.u_point)
            y = brack * np.sin(self.u_point)
            z = self.v_point
        
        return x, y, z
        
    def compute_derivatives(self):
        # compute partial derivatives of the implicit surface function
        
        if TorusModel.geometry_code_t == 0:
            ecc = TorusModel.a_t**2./TorusModel.b_t**2.
            dPdx = 4.*(self.x)**3. + 4.*(self.x)*(self.y)**2. + \
                    4.*ecc*(self.x)*(self.z)**2. - 4.*(self.x)*(TorusModel.R_t)**2. - \
                     4.*(self.x)*(TorusModel.a_t)**2.
            dPdy = 4.*(self.y)**3. + 4.*(self.y)*(self.x)**2. + \
                    4.*ecc*(self.y)*(self.z)**2. - 4.*(self.y)*(TorusModel.R_t)**2. - \
                     4.*(self.y)*(TorusModel.a_t)**2.
            dPdz = 4.*ecc*ecc*(self.z)**3. + 4.*ecc*(self.z)*(self.x)**2. + \
                    4.*ecc*(self.z)*(self.y)**2. + 4.*ecc*(self.z)*(TorusModel.R_t)**2. - \
                     4.*ecc*(self.z)*(TorusModel.a_t)**2.
        elif TorusModel.geometry_code_t == 1:
            s = (self.x*self.x+self.y*self.y)**0.5
            dPdx = -2*(TorusModel.R_t-s)*self.x/TorusModel.a_t**2./s
            dPdy = -2*(TorusModel.R_t-s)*self.y/TorusModel.a_t**2./s
            if self.z > 0.:
                dPdz = -2.*(1-self.z/TorusModel.b_t)/TorusModel.b_t                
            else:
                dPdz = 2.*(self.z+TorusModel.b_t)/TorusModel.b_t**2.
        else:
            brack = TorusModel.r_in_t + (TorusModel.rho_t-TorusModel.r_in_t) \
                        * np.tan(TorusModel.theta_t/180.*np.pi) * abs(self.z) / TorusModel.rho_t
            dPdx = 2.*self.x
            dPdy = 2.*self.y
            if self.z < 0.:
                dPdz = 2.*brack*(TorusModel.rho_t-TorusModel.r_in_t) \
                            * np.tan(TorusModel.theta_t/180.*np.pi)/TorusModel.rho_t
            else:
                dPdz = - 2.*brack*(TorusModel.rho_t-TorusModel.r_in_t) \
                            * np.tan(TorusModel.theta_t/180.*np.pi)/TorusModel.rho_t
        
        return dPdx, dPdy, dPdz
    
    def compute_norm_vector(self):
        # compute the normal vector and its normalization
        
        n_norm = (self.dPdx * self.dPdx + self.dPdy * self.dPdy + \
                       self.dPdz * self.dPdz)**.5
        if TorusModel.geometry_code_t == 1 or TorusModel.geometry_code_t == 2:
            n = [-self.dPdx/n_norm, -self.dPdy/n_norm, \
                      -self.dPdz/n_norm] # normalized
        else:
            n = [self.dPdx/n_norm, self.dPdy/n_norm, \
                      self.dPdz/n_norm] # normalized
            
        return n_norm, n
      
    def projected_vectors(self):
        # compute the I, E vector projections
        
        Ip = []
        Ep = []        
        for i in range(3):
            Ip_i = self.I[i] - self.In*self.n[i]
            Ep_i = self.E[i] - self.En*self.n[i]
            Ip.append(Ip_i)
            Ep.append(Ep_i)
        
        return Ip, Ep
    
    @staticmethod
    def get_norm(vec):
        # compute normalization of any vector
        
        return (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])**.5
        
    def compute_loc_xi(self):
        # computes xi at a given point
        
        if TorusModel.xi0_val_t != 0.:
            loc_xi = TorusModel.xi0_val_t*self.loc_mui*self.a_mu*(TorusModel.r_in_t/self.distance)**(TorusModel.beta_t)
        else:
            print('error: loc_xi should not be computed for this ionization type')
            loc_xi = -1.
        
        return loc_xi
    
    def compute_cosscat_ang(self):
        # compute the cosine of local scattering angle
        
        IE = self.I[0]*self.E[0] + self.I[1]*self.E[1] + \
                    self.I[2]*self.E[2]
        
        cosscat_ang = -IE/(LocalPoint.get_norm(self.I)*LocalPoint.get_norm(self.E))
        
        return cosscat_ang
    
    def compute_loc_iang(self):
        # compute the cosine of angle between the z-axis and the center-local point connector
        
        Iz = self.I[0]*self.z_vec[0] + self.I[1]*self.z_vec[1] + \
                    self.I[2]*self.z_vec[2]
        
        loc_iang = Iz/LocalPoint.get_norm(self.I)
        
        return loc_iang
    
    def compute_loc_mui(self):
        # compute the local angles: cos(delta_i)

        normI = LocalPoint.get_norm(self.I)
        loc_mui = - self.In / normI
        if abs(abs(loc_mui) - 1.) < 0.0001:
            if loc_mui > 1.:
                loc_mui = 1.
            elif loc_mui < -1.:
                loc_mui = -1.
        elif abs(loc_mui - 0.) < 0.0001:
            if loc_mui < 0.:
                loc_mui = 0.

        return loc_mui

    def compute_loc_mue(self):
        # compute the local angles: cos(delta_e)

        loc_mue = self.En
        if abs(abs(loc_mue) - 1.) < 0.0001:
            if loc_mue > 1.:
                loc_mue = 1.
            elif loc_mue < -1.:
                loc_mue = -1.
        elif abs(loc_mue - 0.) < 0.0001:
            if loc_mue < 0.:
                loc_mue = 0.

        return loc_mue

    def compute_loc_Phi(self):        
        # compute the local angles: Phi_e
        
        normIp = LocalPoint.get_norm(self.Ip)
        normEp = LocalPoint.get_norm(self.Ep)
        top = self.Ip[0]*self.Ep[0] + self.Ip[1]*self.Ep[1] + \
                self.Ip[2]*self.Ep[2]
        bottom = normIp * normEp
        argument = top / bottom
        if abs(self.u_point - np.pi/2.) < 0.0001:
            # there was a problem with nan for this, but given the geometry 
            # we are always in forward direction for these u
            loc_Phi = 0.
        elif abs(self.u_point - 3.*np.pi/2.) < 0.0001:
            if argument >= 0.:
                loc_Phi = 0. # forward
            else:
                loc_Phi = np.pi # backward
        else:
            dUEp = (self.Ep[0]*self.dU[0] + self.Ep[1]*self.dU[1] + \
                    self.Ep[2]*self.dU[2])/normEp
            if abs(dUEp + 1.) < 0.0001:
                dUEp = -1.
            elif abs(dUEp - 1.) < 0.0001:
                dUEp = 1.
            decide_angle = np.arccos(dUEp)/np.pi*180.
            if 0. <= decide_angle < 90. or (decide_angle == 90. and abs(argument - 1.)<0.1):
                K = 0.
                sign = 1.
            elif 90. < decide_angle <= 180. or (decide_angle == 90. and abs(argument + 1.)<0.1):
                K = 2.*np.pi
                sign = -1.
            else:
                # debugging
                #print(decide_angle)
                #print(argument)
                #print(np.arccos(argument))
                print("error: local azimuthal angle not found.")
                sys.exit()   
            angl = argument
            if abs(abs(angl) - 1.) < 0.1:
                if angl > 0.:
                    angl = 1.
                else:
                    angl = -1.
            elif abs(angl - 0.) < 0.1:
                angl = 0.                
            loc_Phi = sign*np.arccos(angl) + K
            if loc_Phi == 2*np.pi:
                loc_Phi = 0.

        return loc_Phi/np.pi*180.

    def compute_E_projected(self):        
        # compute the n_p, z_p, e_y with respect to E        
        
        n_p = []
        zp = []
        for i in range(3):            
            zp_i = self.z_vec[i] - self.Ez*self.E[i]
            np_i = self.n[i] - self.En*self.E[i]
            zp.append(zp_i)
            n_p.append(np_i)
         
        npnorm = LocalPoint.get_norm(n_p)
        zpnorm = LocalPoint.get_norm(zp)
        np_normed = []
        zp_normed = []
        for i in range(3):
            np_normed.append(n_p[i]/npnorm)
            zp_normed.append(zp[i]/zpnorm)
            
        y_vec = [self.E[1]*np_normed[2]-self.E[2]*np_normed[1], \
                 self.E[2]*np_normed[0]-self.E[0]*np_normed[2], \
                 self.E[0]*np_normed[1]-self.E[1]*np_normed[0]]
        
        return np_normed, zp_normed, y_vec
    
    def surface(self, u_1, u_2, v_1, v_2):   
        # calculate contribution of the projected local area to the total 
        # output

        if TorusModel.geometry_code_t == 1:
            if self.z > 0.:
                bracket = TorusModel.R_t + TorusModel.a_t*np.cos(self.v_point)
            else:
                bracket = TorusModel.R_t - TorusModel.a_t*np.cos(self.v_point)
            rt = (TorusModel.a_t**2.*np.sin(self.v_point)*np.sin(self.v_point) + \
                  TorusModel.b_t**2.*np.cos(self.v_point)*np.cos(self.v_point))**(0.5)
            surface_area = (v_2-v_1)*(u_2-u_1)*rt*bracket
        elif TorusModel.geometry_code_t == 2:
            s_func = (TorusModel.rho_t-TorusModel.r_in_t) \
                * np.tan(TorusModel.theta_t/180.*np.pi) / TorusModel.rho_t
            if self.z > 0.:
                L_func = TorusModel.r_in_t*(v_2-v_1) + 0.5*s_func*(v_2)**2.-0.5*s_func*(v_1)**2.
            else:
                L_func = TorusModel.r_in_t*(v_2-v_1) - 0.5*s_func*(v_2)**2.+0.5*s_func*(v_1)**2.
            surface_area = L_func*(u_2-u_1)*(1.+s_func*s_func)**(0.5)
        else:
            bracket = TorusModel.R_t + TorusModel.a_t*np.cos(self.v_point)
       
            rt = (TorusModel.a_t**2.*np.sin(self.v_point)*np.sin(self.v_point) + \
                  TorusModel.b_t**2.*np.cos(self.v_point)*np.cos(self.v_point))**(0.5)
            surface_area = (v_2-v_1)*(u_2-u_1)*rt*bracket      
                
        return surface_area
    
    @staticmethod
    def interpolate_incident(three_spec, p0, Psi0):
        # interpolates in the incident polarization state, copies the order 
        # of primary polarization states from the table loading routine
        
        onespec = []
        for en in range(len(three_spec[0])):
            bracket1 = three_spec[1][en]-three_spec[0][en]
            bracket2 = three_spec[2][en]-three_spec[0][en]
            S_final = three_spec[0][en] + p0*(bracket1*np.cos(2.*Psi0) + \
                                              bracket2*np.sin(2.*Psi0))
            onespec.append(S_final)
    
        return onespec

    def interpolate(self, loaded_params, loaded_tables, loaded_tables_neutral, loaded_energies, p0, Psi0):
        # main interpolation function
        
        if TorusModel.inctype_code_t == 2:
            p0 = 0.064*(1.-abs(self.loc_iang))*(1.+16.3*abs(self.loc_iang)+ \
                                       6.2*self.loc_iang*self.loc_iang)/(1.+8.2*abs(self.loc_iang)- \
                                                                  2.1*self.loc_iang*self.loc_iang)
            Psi0 = 0.
        elif TorusModel.inctype_code_t == 3:
            p0 = 0.117*(1.-abs(self.loc_iang))/(1.+3.582*abs(self.loc_iang))
            Psi0 = np.pi/2.
            
        if TorusModel.xi0_val_t > 0.:
            saved_xi = loaded_params[0]
            saved_mui = loaded_params[1]
            saved_mue = loaded_params[2]
            saved_Phi = loaded_params[3]      
    
            # find nearest parameter points in the loaded tables
            flag_found = False
            for m in range(len(saved_mui)-1):
                if saved_mui[m] <= self.loc_mui <= saved_mui[m+1]:
                    mui1 = m
                    mui2 = m+1
                    flag_found = True
            if flag_found == False:
                print('interpolation problem, mu_i out of range: ', self.loc_mui)
                sys.exit()
                
            flag_found = False
            mue_attention = False
            if self.loc_mue <= saved_mue[0]:
                mue1 = 0
                mue2 = 0
                flag_found = True
                mue_attention = True
            elif self.loc_mue >= saved_mue[-1]:
                mue1 = len(saved_mue)-1
                mue2 = len(saved_mue)-1
                mue_attention = True
                flag_found = True
            else:
                for m in range(len(saved_mue)-1):
                    if saved_mue[m] <= self.loc_mue <= saved_mue[m+1]:
                        mue1 = m
                        mue2 = m+1
                        flag_found = True
            if flag_found == False:
                print('interpolation problem, mu_e out of range: ', self.loc_mue)
                sys.exit()    
            
            flag_found = False
            xi_attention = False
            if self.loc_xi <= saved_xi[0]:
                xi1 = 0
                xi2 = 0
                flag_found = True
                xidif = self.loc_xi/saved_xi[0]
                xi_attention = True
            elif self.loc_xi >= saved_xi[-1]:
                xi1 = len(saved_xi)-1
                xi2 = len(saved_xi)-1
                xi_attention = True
                xidif = self.loc_xi/saved_xi[-1]
                flag_found = True
            else:
                for x in range(len(saved_xi)-1):
                    if saved_xi[x] <= self.loc_xi <= saved_xi[x+1]:
                        xi1 = x
                        xi2 = x+1
                        flag_found = True
            if flag_found == False:
                print('interpolation problem, xi out of range: ', self.loc_xi)
                sys.exit()  
            
            flag_found = False
            Phi_change = True
            if 0. <= self.loc_Phi <= saved_Phi[0]:
                Phi1 = len(saved_Phi)-1
                Phi2 = 0
                Phi_add = True
                flag_found = True
            elif saved_Phi[-1] <= self.loc_Phi <= 360.:
                Phi1 = len(saved_Phi)-1
                Phi2 = 0
                Phi_add = False
                flag_found = True
            else:
                Phi_change = False
                for m in range(len(saved_Phi)-1):
                    if saved_Phi[m] <= self.loc_Phi <= saved_Phi[m+1]:
                        Phi1 = m
                        Phi2 = m+1
                        Phi_add = False
                        flag_found = True
            if flag_found == False:
                print('interpolation problem, Phi out of range: ', self.loc_Phi)
                sys.exit()
                
            # for Fwxyz we place: x = mu_i, y = mu_e, z = Phi, w = xi
            # always check this order in the main routine from parameter print out
            F0000_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue1
            F0100_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue1
            F0001_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue1
            F0101_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue1
            F0010_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue2
            F0110_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue2
            F0011_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue2
            F0111_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi1 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue2
            F1000_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue1
            F1100_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue1
            F1001_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue1
            F1101_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue1
            F1010_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue2
            F1110_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue2
            F1011_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue2
            F1111_index = len(saved_mui)*len(saved_Phi)*len(saved_mue)*xi2 + \
                            len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue2
            
            if xi_attention == True:
                # for Cxyz we place: x = mu_i, y = mu_e, z = Phi
                # always check this order in the main routine from parameter print out
                C000_index = F0000_index
                C100_index = F0100_index
                C001_index = F0001_index
                C101_index = F0101_index
                C010_index = F0010_index
                C110_index = F0110_index
                C011_index = F0011_index
                C111_index = F0111_index
                if mue_attention == True:
                    # for Qxz we place: x = mu_i, z = Phi and we don't interpolate in mu_e
                    Q11_index = C000_index
                    Q12_index = C001_index
                    Q21_index = C100_index
                    Q22_index = C101_index
                    
            elif mue_attention == True:
                # for Cxyz we place: x = mu_i, y = xi, z = Phi
                # always check this order in the main routine from parameter print out
                C000_index = F0000_index
                C100_index = F0100_index
                C001_index = F0001_index
                C101_index = F0101_index
                C010_index = F1000_index
                C110_index = F1100_index
                C011_index = F1001_index
                C111_index = F1101_index
            
            IQU_spectra = []
            for iqu in range(len(TorusModel.IQUs_t)):
                if mue_attention == False and xi_attention == False:
                    # for clarity purposes we write out all here rather than 
                    # folding into some functions
                    F0000 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0000_index], p0, Psi0)
                    F0100 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0100_index], p0, Psi0)
                    F0001 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0001_index], p0, Psi0)
                    F0101 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0101_index], p0, Psi0)
                    F0010 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0010_index], p0, Psi0)
                    F0110 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0110_index], p0, Psi0)
                    F0011 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0011_index], p0, Psi0)
                    F0111 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F0111_index], p0, Psi0)
                    F1000 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1000_index], p0, Psi0)
                    F1100 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1100_index], p0, Psi0)
                    F1001 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1001_index], p0, Psi0)
                    F1101 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1101_index], p0, Psi0)
                    F1010 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1010_index], p0, Psi0)
                    F1110 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1110_index], p0, Psi0)
                    F1011 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1011_index], p0, Psi0)
                    F1111 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][F1111_index], p0, Psi0)
                    
                    one_spectrum = []
                    for e in range(len(F0000)):
                        w0 = saved_xi[xi1]
                        w1 = saved_xi[xi2]
                        x0 = saved_mui[mui1]
                        x1 = saved_mui[mui2]
                        y0 = saved_mue[mue1]
                        y1 = saved_mue[mue2]
                        if Phi_change == True:
                            if Phi_add == True:
                                z0 = saved_Phi[Phi1] - 360.
                                z1 = saved_Phi[Phi2]  
                            else:
                                z0 = saved_Phi[Phi1]
                                z1 = saved_Phi[Phi2] + 360.
                        else:
                            z0 = saved_Phi[Phi1]
                            z1 = saved_Phi[Phi2]   
                    
                        # Differences along each axis
                        delta_w = w1 - w0
                        delta_x = x1 - x0
                        delta_y = y1 - y0
                        delta_z = z1 - z0
                        if delta_w == 0. or delta_x == 0. or delta_y == 0. or delta_z == 0.:
                            print('warning two parameters in your tables are '+ \
                                   'the same, unable to interpolate in between')
                            sys.exit()
                        
                        # Assign each coefficient on a new line
                        a0 = F0000[e]
                        a1 = (F1000[e] - F0000[e]) / delta_w
                        a2 = (F0100[e] - F0000[e]) / delta_x
                        a3 = (F0010[e] - F0000[e]) / delta_y
                        a4 = (F0001[e] - F0000[e]) / delta_z
                        
                        a5 = (F1100[e] - F0100[e] - F1000[e] + F0000[e]) / (delta_w * delta_x)
                        a6 = (F1010[e] - F0010[e] - F1000[e] + F0000[e]) / (delta_w * delta_y)
                        a7 = (F1001[e] - F0001[e] - F1000[e] + F0000[e]) / (delta_w * delta_z)
                        a8 = (F0110[e] - F0010[e] - F0100[e] + F0000[e]) / (delta_x * delta_y)
                        a9 = (F0101[e] - F0001[e] - F0100[e] + F0000[e]) / (delta_x * delta_z)
                        a10 = (F0011[e] - F0001[e] - F0010[e] + F0000[e]) / (delta_y * delta_z)
                        
                        a11 = (F1110[e] - F0110[e] - F1010[e] + F0010[e] - F1100[e] + F0100[e] + F1000[e] - F0000[e]) / (delta_w * delta_x * delta_y)
                        a12 = (F1101[e] - F0101[e] - F1001[e] + F0001[e] - F1100[e] + F0100[e] + F1000[e] - F0000[e]) / (delta_w * delta_x * delta_z)
                        a13 = (F1011[e] - F0011[e] - F1001[e] + F0001[e] - F1010[e] + F0010[e] + F1000[e] - F0000[e]) / (delta_w * delta_y * delta_z)
                        a14 = (F0111[e] - F0011[e] - F0101[e] + F0001[e] - F0110[e] + F0010[e] + F0100[e] - F0000[e]) / (delta_x * delta_y * delta_z)
                        
                        a15 = (F1111[e] - F0111[e] - F1011[e] + F0011[e] - F1101[e] + F0101[e] + F1001[e] - F0001[e] - F1110[e] + F0110[e] + F1010[e] - F0010[e] + F1100[e] - F0100[e] - F1000[e] + F0000[e]) / (delta_w * delta_x * delta_y * delta_z)
    
                        
                        final_value = (a0 + a1 * (self.loc_xi - w0) + a2 * (self.loc_mui - x0) + a3 * (self.loc_mue - y0) + a4 * (self.loc_Phi - z0) \
                   + a5 * (self.loc_xi - w0) * (self.loc_mui - x0) + a6 * (self.loc_xi - w0) * (self.loc_mue - y0) + a7 * (self.loc_xi - w0) * (self.loc_Phi - z0) \
                   + a8 * (self.loc_mui - x0) * (self.loc_mue - y0) + a9 * (self.loc_mui - x0) * (self.loc_Phi - z0) + a10 * (self.loc_mue - y0) * (self.loc_Phi - z0) \
                   + a11 * (self.loc_xi - w0) * (self.loc_mui - x0) * (self.loc_mue - y0) + a12 * (self.loc_xi - w0) * (self.loc_mui - x0) * (self.loc_Phi - z0) \
                   + a13 * (self.loc_xi - w0) * (self.loc_mue - y0) * (self.loc_Phi - z0) + a14 * (self.loc_mui - x0) * (self.loc_mue - y0) * (self.loc_Phi - z0) \
                   + a15 * (self.loc_xi - w0) * (self.loc_mui - x0) * (self.loc_mue - y0) * (self.loc_Phi - z0))
    
                                        
                        one_spectrum.append(final_value)
                        
                elif xi_attention == True and mue_attention == True:
                    # we have to do bilinear interpolation if mu_e has 
                    # the same values
                    Q11 = LocalPoint.interpolate_incident( \
                                    loaded_tables[iqu][Q11_index], p0, Psi0)
                    Q12 = LocalPoint.interpolate_incident( \
                                    loaded_tables[iqu][Q12_index], p0, Psi0)
                    Q21 = LocalPoint.interpolate_incident( \
                                    loaded_tables[iqu][Q21_index], p0, Psi0)
                    Q22 = LocalPoint.interpolate_incident( \
                                    loaded_tables[iqu][Q22_index], p0, Psi0)
                    
                    one_spectrum = []
                    for e in range(len(Q11)):
                        x1 = saved_mui[mui1]
                        x2 = saved_mui[mui2]
                        if Phi_change == True:
                            if Phi_add == True:
                                z1 = saved_Phi[Phi1] - 360.
                                z2 = saved_Phi[Phi2]  
                            else:
                                z1 = saved_Phi[Phi1]
                                z2 = saved_Phi[Phi2] + 360.
                        else:
                            z1 = saved_Phi[Phi1]
                            z2 = saved_Phi[Phi2]   
                            
                        bottom = (x2-x1)*(z2-z1)    
                        if bottom == 0.:
                            print('warning two parameters in your tables are '+ \
                                  'the same, unable to interpolate in between')
                            sys.exit()
                            
                        # bi-linear interpolation coefficients
                        w11 = (x2-self.loc_mui)*(z2-self.loc_Phi)
                        w12 = (x2-self.loc_mui)*(self.loc_Phi-z1)
                        w21 = (self.loc_mui-x1)*(z2-self.loc_Phi)
                        w22 = (self.loc_mui-x1)*(self.loc_Phi-z1)
                        
                        final_value = Q11[e]*w11 + Q12[e]*w12 + Q21[e]*w21 + \
                                        Q22[e]*w22                    
                        one_spectrum.append(xidif*final_value/bottom)
                        
                else:
                    # for clarity purposes we write out all here rather than 
                    # folding into some functions
                    C000 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C000_index], p0, Psi0)
                    C100 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C100_index], p0, Psi0)
                    C010 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C010_index], p0, Psi0)
                    C110 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C110_index], p0, Psi0)
                    C001 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C001_index], p0, Psi0)
                    C101 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C101_index], p0, Psi0)
                    C011 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C011_index], p0, Psi0)
                    C111 = LocalPoint.interpolate_incident( \
                                        loaded_tables[iqu][C111_index], p0, Psi0)
                    
                    one_spectrum = []
                    for e in range(len(C000)):
                        x0 = saved_mui[mui1]
                        x1 = saved_mui[mui2]
                        if mue_attention == True:
                            y0 = saved_xi[xi1]
                            y1 = saved_xi[xi2]
                        else:
                            y0 = saved_mue[mue1]
                            y1 = saved_mue[mue2]
                        if Phi_change == True:
                            if Phi_add == True:
                                z0 = saved_Phi[Phi1] - 360.
                                z1 = saved_Phi[Phi2]  
                            else:
                                z0 = saved_Phi[Phi1]
                                z1 = saved_Phi[Phi2] + 360.
                        else:
                            z0 = saved_Phi[Phi1]
                            z1 = saved_Phi[Phi2]   
                        
                        bottom = (x0-x1)*(y0-y1)*(z0-z1)
                        if bottom == 0.:
                            print('warning two parameters in your tables are '+ \
                                   'the same, unable to interpolate in between')
                            sys.exit()
                   
                        # tri-linear interpolation coefficients
                        a0 = -C000[e]*x1*y1*z1 + C001[e]*x1*y1*z0 + \
                                C010[e]*x1*y0*z1 - C011[e]*x1*y0*z0 \
                                +C100[e]*x0*y1*z1 - C101[e]*x0*y1*z0 - \
                                    C110[e]*x0*y0*z1 + C111[e]*x0*y0*z0
                        a1 = C000[e]*y1*z1 - C001[e]*y1*z0 - C010[e]*y0*z1 + \
                            C011[e]*y0*z0 - C100[e]*y1*z1 + C101[e]*y1*z0 + \
                                C110[e]*y0*z1 - C111[e]*y0*z0
                        a2 = C000[e]*x1*z1 - C001[e]*x1*z0 - C010[e]*x1*z1 + \
                                C011[e]*x1*z0 - C100[e]*x0*z1 + C101[e]*x0*z0 + \
                                    C110[e]*x0*z1 - C111[e]*x0*z0
                        a3 = C000[e]*x1*y1 - C001[e]*x1*y1 - C010[e]*x1*y0 + \
                               C011[e]*x1*y0 - C100[e]*x0*y1 + C101[e]*x0*y1 + \
                                   C110[e]*x0*y0 - C111[e]*x0*y0
                        a4 = -C000[e]*z1 + C001[e]*z0 + C010[e]*z1 - \
                                C011[e]*z0 + C100[e]*z1 - C101[e]*z0 - \
                                    C110[e]*z1 + C111[e]*z0
                        a5 = -C000[e]*y1 + C001[e]*y1 + C010[e]*y0 - \
                                C011[e]*y0 + C100[e]*y1 - C101[e]*y1 - \
                                    C110[e]*y0 + C111[e]*y0
                        a6 = -C000[e]*x1 + C001[e]*x1 + C010[e]*x1 - \
                                C011[e]*x1 + C100[e]*x0 - C101[e]*x0 - \
                                    C110[e]*x0 + C111[e]*x0
                        a7 = C000[e] - C001[e] - C010[e] + C011[e] - \
                                C100[e] + C101[e] + C110[e] - C111[e]
                        
                        if mue_attention == True:
                            final_value = a0 + a1*self.loc_mui + a2*self.loc_xi + \
                                            a3*self.loc_Phi + \
                                            a4*self.loc_mui*self.loc_xi + \
                                            a5*self.loc_mui*self.loc_Phi + \
                                            a6*self.loc_xi*self.loc_Phi + \
                                            a7*self.loc_mui*self.loc_xi*self.loc_Phi
                            one_spectrum.append(final_value/bottom)
                        else:
                            final_value = a0 + a1*self.loc_mui + a2*self.loc_mue + \
                                            a3*self.loc_Phi + \
                                            a4*self.loc_mui*self.loc_mue + \
                                            a5*self.loc_mui*self.loc_Phi + \
                                            a6*self.loc_mue*self.loc_Phi + \
                                            a7*self.loc_mui*self.loc_mue*self.loc_Phi
                            one_spectrum.append(xidif*final_value/bottom)
     
                IQU_spectra.append(one_spectrum)
                
        elif TorusModel.xi0_val_t == 0.:
            saved_mui = loaded_params[1]
            saved_mue = loaded_params[2]
            saved_Phi = loaded_params[3]      
    
            # find nearest parameter points in the loaded tables
            flag_found = False
            for m in range(len(saved_mui)-1):
                if saved_mui[m] <= self.loc_mui <= saved_mui[m+1]:
                    mui1 = m
                    mui2 = m+1
                    flag_found = True
            if flag_found == False:
                print('interpolation problem, mu_i out of range: ', self.loc_mui)
                sys.exit()
                
            flag_found = False
            mue_attention = False
            if self.loc_mue <= saved_mue[0]:
                mue1 = 0
                mue2 = 0
                flag_found = True
                mue_attention = True
            elif self.loc_mue >= saved_mue[-1]:
                mue1 = len(saved_mue)-1
                mue2 = len(saved_mue)-1
                mue_attention = True
                flag_found = True
            else:
                for m in range(len(saved_mue)-1):
                    if saved_mue[m] <= self.loc_mue <= saved_mue[m+1]:
                        mue1 = m
                        mue2 = m+1
                        flag_found = True
            if flag_found == False:
                print('interpolation problem, mu_e out of range: ', self.loc_mue)
                sys.exit()    
            
            flag_found = True
            xi_attention = True
            xi1 = 0
            xi2 = 0
            xidif = 1.
            
            flag_found = False
            Phi_change = True
            if 0. <= self.loc_Phi <= saved_Phi[0]:
                Phi1 = len(saved_Phi)-1
                Phi2 = 0
                Phi_add = True
                flag_found = True
            elif saved_Phi[-1] <= self.loc_Phi <= 360.:
                Phi1 = len(saved_Phi)-1
                Phi2 = 0
                Phi_add = False
                flag_found = True
            else:
                Phi_change = False
                for m in range(len(saved_Phi)-1):
                    if saved_Phi[m] <= self.loc_Phi <= saved_Phi[m+1]:
                        Phi1 = m
                        Phi2 = m+1
                        Phi_add = False
                        flag_found = True
            if flag_found == False:
                print('interpolation problem, Phi out of range: ', self.loc_Phi)
                sys.exit()
                
            # for Fwxyz we place: x = mu_i, y = mu_e, z = Phi, w = xi
            # always check this order in the main routine from parameter print out
            F0000_index = len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue1
            F0100_index = len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue1
            F0001_index = len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue1
            F0101_index = len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue1
            F0010_index = len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi1 + mue2
            F0110_index = len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi1 + mue2
            F0011_index = len(saved_Phi)*len(saved_mue)*mui1 + \
                            len(saved_mue)*Phi2 + mue2
            F0111_index = len(saved_Phi)*len(saved_mue)*mui2 + \
                            len(saved_mue)*Phi2 + mue2
            
            if xi_attention == True:
                # for Cxyz we place: x = mu_i, y = mu_e, z = Phi
                # always check this order in the main routine from parameter print out
                C000_index = F0000_index
                C100_index = F0100_index
                C001_index = F0001_index
                C101_index = F0101_index
                C010_index = F0010_index
                C110_index = F0110_index
                C011_index = F0011_index
                C111_index = F0111_index
                if mue_attention == True:
                    # for Qxz we place: x = mu_i, z = Phi and we don't interpolate in mu_e
                    Q11_index = C000_index
                    Q12_index = C001_index
                    Q21_index = C100_index
                    Q22_index = C101_index
            
            IQU_spectra = []
            for iqu in range(len(TorusModel.IQUs_t)):                    
                if xi_attention == True and mue_attention == True:
                    # we have to do bilinear interpolation if mu_e has 
                    # the same values
                    Q11 = LocalPoint.interpolate_incident( \
                                    loaded_tables_neutral[iqu][Q11_index], p0, Psi0)
                    Q12 = LocalPoint.interpolate_incident( \
                                    loaded_tables_neutral[iqu][Q12_index], p0, Psi0)
                    Q21 = LocalPoint.interpolate_incident( \
                                    loaded_tables_neutral[iqu][Q21_index], p0, Psi0)
                    Q22 = LocalPoint.interpolate_incident( \
                                    loaded_tables_neutral[iqu][Q22_index], p0, Psi0)
                    
                    one_spectrum = []
                    for e in range(len(Q11)):
                        x1 = saved_mui[mui1]
                        x2 = saved_mui[mui2]
                        if Phi_change == True:
                            if Phi_add == True:
                                z1 = saved_Phi[Phi1] - 360.
                                z2 = saved_Phi[Phi2]  
                            else:
                                z1 = saved_Phi[Phi1]
                                z2 = saved_Phi[Phi2] + 360.
                        else:
                            z1 = saved_Phi[Phi1]
                            z2 = saved_Phi[Phi2]   
                            
                        bottom = (x2-x1)*(z2-z1)    
                        if bottom == 0.:
                            print('warning two parameters in your tables are '+ \
                                  'the same, unable to interpolate in between')
                            sys.exit()
                            
                        # bi-linear interpolation coefficients
                        w11 = (x2-self.loc_mui)*(z2-self.loc_Phi)
                        w12 = (x2-self.loc_mui)*(self.loc_Phi-z1)
                        w21 = (self.loc_mui-x1)*(z2-self.loc_Phi)
                        w22 = (self.loc_mui-x1)*(self.loc_Phi-z1)
                        
                        final_value = Q11[e]*w11 + Q12[e]*w12 + Q21[e]*w21 + \
                                        Q22[e]*w22                    
                        one_spectrum.append(xidif*final_value/bottom)
                        
                else:
                    # for clarity purposes we write out all here rather than 
                    # folding into some functions
                    C000 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C000_index], p0, Psi0)
                    C100 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C100_index], p0, Psi0)
                    C010 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C010_index], p0, Psi0)
                    C110 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C110_index], p0, Psi0)
                    C001 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C001_index], p0, Psi0)
                    C101 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C101_index], p0, Psi0)
                    C011 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C011_index], p0, Psi0)
                    C111 = LocalPoint.interpolate_incident( \
                                        loaded_tables_neutral[iqu][C111_index], p0, Psi0)
                    
                    one_spectrum = []
                    for e in range(len(C000)):
                        x0 = saved_mui[mui1]
                        x1 = saved_mui[mui2]
                        y0 = saved_mue[mue1]
                        y1 = saved_mue[mue2]
                        if Phi_change == True:
                            if Phi_add == True:
                                z0 = saved_Phi[Phi1] - 360.
                                z1 = saved_Phi[Phi2]  
                            else:
                                z0 = saved_Phi[Phi1]
                                z1 = saved_Phi[Phi2] + 360.
                        else:
                            z0 = saved_Phi[Phi1]
                            z1 = saved_Phi[Phi2]   
                        
                        bottom = (x0-x1)*(y0-y1)*(z0-z1)
                        if bottom == 0.:
                            print('warning two parameters in your tables are '+ \
                                   'the same, unable to interpolate in between')
                            sys.exit()
                   
                        # tri-linear interpolation coefficients
                        a0 = -C000[e]*x1*y1*z1 + C001[e]*x1*y1*z0 + \
                                C010[e]*x1*y0*z1 - C011[e]*x1*y0*z0 \
                                +C100[e]*x0*y1*z1 - C101[e]*x0*y1*z0 - \
                                    C110[e]*x0*y0*z1 + C111[e]*x0*y0*z0
                        a1 = C000[e]*y1*z1 - C001[e]*y1*z0 - C010[e]*y0*z1 + \
                            C011[e]*y0*z0 - C100[e]*y1*z1 + C101[e]*y1*z0 + \
                                C110[e]*y0*z1 - C111[e]*y0*z0
                        a2 = C000[e]*x1*z1 - C001[e]*x1*z0 - C010[e]*x1*z1 + \
                                C011[e]*x1*z0 - C100[e]*x0*z1 + C101[e]*x0*z0 + \
                                    C110[e]*x0*z1 - C111[e]*x0*z0
                        a3 = C000[e]*x1*y1 - C001[e]*x1*y1 - C010[e]*x1*y0 + \
                               C011[e]*x1*y0 - C100[e]*x0*y1 + C101[e]*x0*y1 + \
                                   C110[e]*x0*y0 - C111[e]*x0*y0
                        a4 = -C000[e]*z1 + C001[e]*z0 + C010[e]*z1 - \
                                C011[e]*z0 + C100[e]*z1 - C101[e]*z0 - \
                                    C110[e]*z1 + C111[e]*z0
                        a5 = -C000[e]*y1 + C001[e]*y1 + C010[e]*y0 - \
                                C011[e]*y0 + C100[e]*y1 - C101[e]*y1 - \
                                    C110[e]*y0 + C111[e]*y0
                        a6 = -C000[e]*x1 + C001[e]*x1 + C010[e]*x1 - \
                                C011[e]*x1 + C100[e]*x0 - C101[e]*x0 - \
                                    C110[e]*x0 + C111[e]*x0
                        a7 = C000[e] - C001[e] - C010[e] + C011[e] - \
                                C100[e] + C101[e] + C110[e] - C111[e]
                        
                        final_value = a0 + a1*self.loc_mui + a2*self.loc_mue + \
                                        a3*self.loc_Phi + \
                                        a4*self.loc_mui*self.loc_mue + \
                                        a5*self.loc_mui*self.loc_Phi + \
                                        a6*self.loc_mue*self.loc_Phi + \
                                        a7*self.loc_mui*self.loc_mue*self.loc_Phi

                        # erg to keV factor
                        ergkeV = 624150647.99632
                        xi_0 = 1.
                        E_min = 10.**(-1.1)
                        E_max = 10.**(2.4)
                        if float(TorusModel.Gamma_t) == 2.:
                            scale_factor = np.log(E_max/E_min)
                        else:
                            scale_factor = (E_max**(2.-float(TorusModel.Gamma_t))- \
                                            E_min**(2.-float(TorusModel.Gamma_t)))/ \
                                            (2.-float(TorusModel.Gamma_t))
                        N_0 = ergkeV*xi_0*10.**(15.)/(scale_factor*4.*np.pi)
                        one_spectrum.append(xidif*final_value/bottom*N_0*self.loc_mui*self.a_mu*(TorusModel.r_in_t/self.distance)**(TorusModel.beta_t))

                IQU_spectra.append(one_spectrum)
        
        return IQU_spectra
    
    def Psifinal_from_Psiorig(self, Psi_orig, Ainverse):
        # polarization angle transformation according to the frame of reference
        
        B = [np.cos(Psi_orig), np.sin(Psi_orig), \
                     0.] # (w_loc in the paper description)
        w = []
        for i in range(3):
            one_row = Ainverse[i][0]*B[0]+Ainverse[i][1]*B[1]+ \
                        Ainverse[i][2]*B[2]
            w.append(one_row)            
        w_k = [self.E[2]*w[1]-self.E[1]*w[2], \
               self.E[0]*w[2]-self.E[2]*w[0], \
               self.E[1]*w[0]-self.E[0]*w[1]]
        w_k_norm = LocalPoint.get_norm(w_k)
        w_norm = LocalPoint.get_norm(w)
        
        cos_fin = (self.zp[0]*w[0]+self.zp[1]*w[1]+self.zp[2]*w[2])/w_norm
        if abs(1.-cos_fin) < 0.0001:
            cos_fin = 1.
        if abs(-1.-cos_fin) < 0.0001:
            cos_fin = -1.    
        argu = (self.zp[0]*w_k[0]+self.zp[1]*w_k[1]+ \
                    self.zp[2]*w_k[2])/w_k_norm
        if abs(1.-argu) < 0.0001:
            argu = 1.
        if abs(-1.-argu) < 0.0001:
            argu = -1.
        
        angle_dec = np.arccos(argu)
        if angle_dec <= np.pi/2.:
            Psi_final = np.arccos(cos_fin)
        else:
            Psi_final = -np.arccos(cos_fin)
        
        return Psi_final
    
    def rotate(self, IQU_spectra):
        # rotate the polarization vector from local coordinates to global
        # this routine can be used similarly if rotation on the input is 
        # needed, if broken symmetry
        
        IQU_spectra_rotated = [[],[],[]]
        
        # Calculating the inverse of the matrix (denoted as B in the paper 
        # description)
        A = np.array([self.n_p,self.y_vec,self.E])
        Ainverse = np.linalg.inv(A)
        
        for e in range(len(IQU_spectra[0])):
            one_Q = IQU_spectra[1][e]
            one_U = IQU_spectra[2][e]
            
            Psi_orig = 0.5*(np.arctan2(one_U,one_Q))                                
            if abs(Psi_orig) < 0.0001:
                Psi_orig = 0.
            if not (0. <= Psi_orig < np.pi):
                Psi1 = Psi_orig
                Psi2 = Psi_orig
                n = 0
                while (Psi1 < 0. or Psi1 >= np.pi) and (Psi2 < 0. or \
                                                        Psi2 >= np.pi):
                    Psi1 = Psi1 + np.pi
                    Psi2 = Psi2 - np.pi
                    n+=1
                    if n >= 200:
                        print(n, Psi1, Psi2, Psi_orig, one_Q, one_U)
                        sys.exit()
                if (0. <= Psi1 < np.pi):
                    Psi_orig = Psi1
                elif (0. <= Psi2 < np.pi):
                    Psi_orig = Psi2
                else:
                    print('error in getting local Psi')
                    sys.exit()
            
            Psi_final = self.Psifinal_from_Psiorig(Psi_orig, Ainverse)
        
            sqrtQU = (one_Q*one_Q+one_U*one_U)**.5
            final_Q = sqrtQU*np.cos(2.*Psi_final)
            final_U = sqrtQU*np.sin(2.*Psi_final)
        
            IQU_spectra_rotated[0].append(IQU_spectra[0][e])
            IQU_spectra_rotated[1].append(final_Q)
            IQU_spectra_rotated[2].append(final_U)       
        
        return IQU_spectra_rotated

# if executed as main file then
# run the main function and exit with returned error code
if __name__ == "__main__": 
    print("Provides classes to help to create a table model for XSPEC. "+ \
          "See the documentation for a usage guide.")
    sys.exit(0)
#end if
