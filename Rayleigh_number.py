import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mixture_properties as mp

file_path = r"C:\Users\abdel\Desktop\figures\figures\recovered_temperatures.csv"
file = pd.read_csv(file_path)
df = pd.DataFrame(file)
print(df.head())

T_L = df['T_Pred_Pt1_degC'].to_numpy()
T_H = df['T_Pred_Pt2_degC'].to_numpy()

Delta_T = T_H - T_L
T_avg = (T_H + T_L) / 2
L = 0.031 # m
c_p = mp.c_p_mixture(0.9, T_avg) # J/kg*K
k = mp.k_mixture(0.9, T_avg) # W/m*K
eta = mp.dyn_visc_mixture(0.9, T_avg) # Pa*s
rho_avg = mp.density_mixture(0.9, T_avg) # kg/m^3
rho_H = mp.density_mixture(0.9, T_H) # kg/m^3
rho_L = mp.density_mixture(0.9, T_L) # kg/m^3
g = 9.81 # m/s^2

Ra = - (c_p * rho_avg * g * L**3 * (rho_H - rho_L)) / (eta * k)

plt.figure(figsize=(10, 6))
plt.plot(T_avg, Ra)
plt.title('Rayleigh Number vs Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Rayleigh Number')
plt.tight_layout()
plt.grid()
plt.show()
