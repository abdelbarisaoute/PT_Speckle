"""
Glycerol-Water Mixture Properties Calculator
============================================
Author: [Abdelbari Saoutelhak]
Email: [abdelbarisaoutelhak@gmail.com]
Date: [04/04/2026]
Version: 1.2.0
License: MIT

Description:
This script calculates the density, dynamic viscosity, kinematic viscosity, 
volume contraction, and estimated refractive index of glycerol-water mixtures.
It takes the temperature (°C) and the volume fraction of glycerol (0.0 to 1.0) 
as inputs and handles both scalar and array inputs. Includes Lorentz-Lorenz calculations.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from uncertainties import ufloat
from uncertainties import unumpy
from CoolProp.CoolProp import PropsSI

_df_k = pd.read_csv(r"C:\Users\abdel\Desktop\data\thermal_conductivity_wg.csv", index_col=0)
_df_k.columns = _df_k.columns.astype(float)

# Extract coordinates
_mass_frac_axis = _df_k.index.values / 100.0  # [0.0, 0.1, 0.15, ...]
_temp_axis = _df_k.columns.values             # [10.0, 20.0, ...]

# Convert the entire data table to SI units (W/m*K)
_k_si_values = _df_k.values * 418.4

# 2. Create the 2D Interpolator
# We use 'cubic' for smooth curves or 'linear' for straight connections
_k_interp = RegularGridInterpolator(
    (_mass_frac_axis, _temp_axis), 
    _k_si_values, 
    method='cubic', 
    bounds_error=False, 
    fill_value=None
)


pd.set_option('display.precision', 10)

#Setting files for glycerol water mixture heat capacity interpolation


def _to_array(x):
    return np.atleast_1d(np.asarray(x, dtype=float))

def _wrangler(x, y):
    if np.isscalar(x) and np.isscalar(y):
        return x, y
    else:
        x = _to_array(x)
        y = _to_array(y)
        if x.size == 1 and y.size == 1:
            return x.ravel(), y.ravel()
        X, Y = np.meshgrid(x, y)
        return X.ravel(), Y.ravel()
    
def _wrangler3(x, y, z):
    x = _to_array(x)
    y = _to_array(y)
    z = _to_array(z)
    if x.size == 1 and y.size == 1 and z.size == 1:
        return x.ravel(), y.ravel(), z.ravel()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X.ravel(), Y.ravel(), Z.ravel()

# The temperature dependant density of pure glycerol
def density_glyc(temperature):
    rho_g = 1273 - 0.612 * temperature
    return rho_g

# The temperature dependant density of pure water
def density_water(temperature):
    rho_w = 1000 * (1 - np.abs((temperature - 3.98) / 615)**1.71)
    return rho_w

# The dynamic viscosity of the water in Pa*s
def dyn_visc_water(temperature):
    visc = 1.790 * np.exp( ((-1230 - temperature) * temperature) / (36100 + 360*temperature) ) * 0.001
    return visc

# The dynamic viscosity of the glycerol in Pa*s
def dyn_visc_glyc(temperature):
    visc = 12100 * np.exp( ((-1233 + temperature) * temperature) / (9900 + 70*temperature) ) * 0.001
    return visc

# The mass fraction of glycerol in the mixture
def mass_fraction_glyc(volume_fraction_glycerol, temperature):
    w_g = 1 / (1+((1-volume_fraction_glycerol)/volume_fraction_glycerol)*(density_water(temperature)/density_glyc(temperature))) 
    return w_g

# The volume contraction of the mixture as a percentage (%)
def kappa(volume_fraction_glycerol, temperature):
    A = 1.78e-6 * temperature**2 - 1.82e-4 * temperature + 1.41e-2
    kappa = 1 + A * np.sin(np.radians(mass_fraction_glyc(volume_fraction_glycerol, temperature)**1.31 * 180))**0.81
    return kappa

# The density of the mixture in kg/m^3
def density_mixture(volume_fraction_glycerol, temperature):
    x = density_glyc(temperature) / density_water(temperature) * (1/mass_fraction_glyc(volume_fraction_glycerol, temperature) - 1)
    rho_s  = kappa(volume_fraction_glycerol, temperature) * (  density_water(temperature) + (density_glyc(temperature)-density_water(temperature)) / (1+x) )
    return rho_s

# The dynamic viscosity of the mixture in Pa*s
def dyn_visc_mixture(volume_fraction_glycerol, temperature):
    mu_g = dyn_visc_glyc(temperature)
    mu_w = dyn_visc_water(temperature)
    A = np.log(mu_w/mu_g)
    a = 0.705 - 0.0017*temperature
    b = (4.9 + 0.036*temperature)*a**2.5
    c_m = mass_fraction_glyc(volume_fraction_glycerol, temperature)
    alpha = 1 - c_m + (a*b*c_m*(1-c_m))/(a*c_m + b*(1-c_m))
    return mu_g * np.exp(A * alpha)

# The kinematic viscosity of the mixture in m^2/s
def kin_visc_mixture(volume_fraction_glycerol, temperature):
    dyn_visc = dyn_visc_mixture(volume_fraction_glycerol, temperature)
    density = density_mixture(volume_fraction_glycerol, temperature)
    return dyn_visc / density

# Contraction of the mixture as a percentage (%)
def volume_contraction_mixture(volume_fraction_glycerol, temperature):
    return (1 - (1 / kappa(volume_fraction_glycerol, temperature))) * 100

# Refractive index of pure water
def n_water(lamda, temperature):
    T_ref = 273.15  # Reference temperature in Kelvin
    rho_ref = 1000  # Reference density in kg/m^3
    lamda_ref = 0.589  # Reference wavelength in micrometers
    temperature_bar = (temperature + 273.15) / T_ref
    lamda_bar = (lamda / 1000.0) / lamda_ref
    rho_w = density_water(temperature)
    rho_bar = rho_w / rho_ref  

    # coefficients (IAPWS)
    a0 = 0.244257733
    a1 = 9.74634476e-3
    a2 = -3.73234996e-3
    a3 = 2.68678472e-4
    a4 = 1.58920570e-3
    a5 = 2.45934259e-3
    a6 = 0.900704920
    a7 = -1.66626219e-2

    # resonance wavelengths (µm)
    lamda_uv = 0.229202
    lamda_ir = 5.432937

    R = (a0 
         + a1 * rho_bar 
         + a2 * temperature_bar 
         + a3 * temperature_bar * lamda_bar**2 
         + a4 * lamda_bar**(-2) 
         + a5 / (lamda_bar**2 - lamda_uv**2) 
         + a6 / (lamda_bar**2 - lamda_ir**2) 
         + a7 * rho_bar**2) * rho_bar
    
    n = np.sqrt((2 * R + 1) / (1 - R))
    return np.round(n, 4)

# Refractive index of pure glycerol
def n_glyc(lamda, temperature):
    lamda = lamda / 1000.0  # Convert nm to µm

    # Coefficients for glycerol 
    A = 1.6062
    B_ir = 0.0622
    c_ir = 7.99
    B_uv = 0.53803
    c_uv = 0.0181
    A_t, B_t, c_t = -2.395e-4, -6.2e-6, 0.18

    n_sqr_20 = A + B_ir * lamda**2 / (lamda**2 - c_ir) + B_uv * lamda**2 / (lamda**2 - c_uv)
    dn_dt = A_t + B_t / (lamda**2 - c_t) 
    n_lambda_T = np.sqrt(n_sqr_20) + dn_dt * (temperature - 20)
    return np.round(n_lambda_T, 4)

# Lorentz-Lorenz calculation for the mixture's refractive index
def R_mixture(volume_fraction_glycerol, temperature, lamda):
    n_w = n_water(lamda, temperature)
    n_g = n_glyc(lamda, temperature)
    d_w = density_water(temperature)
    d_g = density_glyc(temperature)
    w_g = mass_fraction_glyc(volume_fraction_glycerol, temperature)
    w_w = 1 - w_g
    return w_w * (n_w**2 - 1) / ((n_w**2 + 2) * d_w) + w_g * (n_g**2 - 1) / ((n_g**2 + 2) * d_g)

# Refractive index of the mixture using Lorentz-Lorenz equation
def n_mixture(volume_fraction_glycerol, temperature, lamda):
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))  # ← add this
    volume_fraction_glycerol = np.atleast_1d(np.asarray(volume_fraction_glycerol, dtype=float)) 
    n_w_val = n_water(lamda, temperature)
    n_g_val = n_glyc(lamda, temperature)
    d_w_val = density_water(temperature)
    d_g_val = density_glyc(temperature)
    d_mix_val = density_mixture(volume_fraction_glycerol, temperature)
    
    n_w = unumpy.uarray(n_w_val, 0.000015) 
    n_g = unumpy.uarray(n_g_val, 0.0003)   
    d_w = unumpy.uarray(d_w_val, 0.01)     
    d_g = unumpy.uarray(d_g_val, 0.1)      
    d_mix = unumpy.uarray(d_mix_val, d_mix_val * 0.0007) # 0.07% error
    
    w_g = (volume_fraction_glycerol * d_g) / (volume_fraction_glycerol * d_g + (1 - volume_fraction_glycerol) * d_w)
    w_w = 1 - w_g
    
    R = w_w * (n_w**2 - 1) / ((n_w**2 + 2) * d_w) + w_g * (n_g**2 - 1) / ((n_g**2 + 2) * d_g)
    
    n_mix = unumpy.sqrt((2 * (R * d_mix) + 1) / (1 - R * d_mix))
    
    nominals = unumpy.nominal_values(n_mix)
    std_devs = unumpy.std_devs(n_mix)
        
    return nominals

# The temperature dependant thermal conductivity of pure water (W/m*K)
def k_water(temperature):
    temperature = temperature + 273.15  # Convert to Kelvin
    return PropsSI('conductivity', 'T', temperature, 'P', 101325, 'Water')

# The temperature dependant thermal conductivity of pure glycerol (W/m*K)
def k_glyc(temperature):
    temperature = temperature + 273.15  # Convert to Kelvin
    k = 402.42 - 0.81155 * temperature + 0.0013903 * temperature**2
    return k * 10**-3

# Constant c for k_mixture calculation
def c(volume_fraction_glycerol, temperature):
    w_g = mass_fraction_glyc(volume_fraction_glycerol, temperature)
    Temp = temperature + 273.15
    c_val = -0.099184 + (0.141481 * w_g) + (0.000666 * Temp) + (0.301722 * w_g**2) + (-0.002144 * w_g * Temp)
    return c_val

# The thermal conductivity of the mixture in W/m*K
def k_mixture(volume_fraction_glycerol, temperature):
    # Convert incoming volume fraction to mass fraction (as per your CSV index)
    w_g = mass_fraction_glyc(volume_fraction_glycerol, temperature)
    
    # Handle scalar or array inputs for the calculator
    w_g_arr, t_arr = np.broadcast_arrays(w_g, temperature)
    pts = np.column_stack((w_g_arr.ravel(), t_arr.ravel()))
    
    # Perform interpolation
    result = _k_interp(pts)
    
    return result.reshape(w_g_arr.shape) if w_g_arr.ndim > 0 else float(result)

# The heat capacity of glycerol from 298 to 383 K (J/kg*K)
def c_p_glyc(temperature):
    molar_mass_kg = 0.0920938
    temp_arr = temperature + 273.15  # Convert to Kelvin without modifying input in place
    c_p_molar = 90.983 + 0.4335 * temp_arr
    c_p_mass = c_p_molar / molar_mass_kg
    return c_p_mass

def c_p_water(temperature):
    temperature = temperature + 273.15  # Convert to Kelvin
    P = 101325  # Pa
    return PropsSI('C', 'T', temperature, 'P', P, 'Water')

# The heat capacity of the water-glycerol mixture in J/kg*K
def c_p_mixture(volume_fraction_glycerol, temperature):
    w_g = mass_fraction_glyc(volume_fraction_glycerol, temperature)
    w_w = 1 - w_g
    c_p_w = c_p_water(temperature)
    c_p_g = c_p_glyc(temperature)
    return w_g * c_p_g + w_w * c_p_w

print(PropsSI('conductivity', 'T', 25+273.15, 'P', 101325, 'Water'))