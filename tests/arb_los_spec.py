import os
os.environ["XUVTOP"] = "/home/ivan/Study/Astro/solar/xuv/CHIANTI_10.0.2_database"
from cProfile import label
from colorsys import yiq_to_rgb
import math
from turtle import color
import numpy as np
from scipy.optimize import curve_fit

import yt

import pickle
import gc
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ChiantiPy
import ChiantiPy.core as ch
# Astropy
from astropy import units as u
from astropy import constants as const
from astropy.modeling import models, fitting

#%%
(1300.08 * u.AA).to(u.Hz, equivalencies=u.spectral())
Kb = 1.38e-16 # (erg deg^-1)
Mp = 1.6726e-24 # (g)
c_light = 2.99792458e+10 # (cm/s)

# iron mass
mfe = 56.0*Mp

# frequency
nu0_1354 = 2.2139937e+15 # (s^-1)
nu0_192 = 1.5610938e+16 # (s^-1)

#%%
#
# 1.1
#
l_char = 1.5e+10 #(cm)
rho_char = 2.5e+8 #(m^-3)
te_char = 1.13e8 #(K)
v_char = 1.366e8 #(cm/s)
time_char = 109.8 #(s)

# set scaled temperature and velocity
te_scale_list = np.array([0.8, 1.0, 1.2])

nsample = 2001

#%%
# restore from the saved pickle files running the above commands
[fe21, fe24] = pickle.load(open("./data_chianti_fe/fe21_24.p", "rb"))
#
# function to get g(tc) for both Fe21 and Fe24 lines
#
def func_g(tc, fe21, fe24):
    gt_1354 = np.interp(tc, fe21.Gofnt['temperature'], fe21.Gofnt['gofnt'])
    gt_192  = np.interp(tc, fe24.Gofnt['temperature'], fe24.Gofnt['gofnt'])
    return gt_1354, gt_192


def func_phi_nu(nu, nu0, tc, vs):
    # thermal speed and boradening
    v_thermal = np.sqrt(2.0 * Kb * tc / mfe)
    delt_nud = (nu / c_light) * v_thermal

    # frequency shift
    delt_nu = nu - nu0

    # Relative velocity of iron
    v_ratio = vs / c_light

    part_head = 1. / (math.sqrt(math.pi) * delt_nud)
    part_exp_in = -((delt_nu + nu0 * v_ratio) / delt_nud) ** 2

    # phi_nu
    phi_nu = part_head * np.exp(part_exp_in)

    return phi_nu
#
# func: intensity_fe21
#
def func_intensity_fe21(lines, v_sample, te_sample, rho_sample, rho_char_in, te_char_in, v_char_in):
    #
    # (old 4.1 Fe XX1 all)
    #
    n_wv = 201  # set resolution of the wavelength
    wv_arr = np.linspace(1352, 1356, n_wv)
    I_nu = np.zeros(n_wv)
    lines_cm = lines * l_char

    for k in range(n_wv):
        wv_c = wv_arr[k]
        nu_c = c_light / (wv_c * 1.0e-8)  # wavelength: 1 AA -> 1.0e-8 cm

        I_1d = np.zeros(nsample)
        for j in range(nsample):
            n_e = rho_sample[j] #* rho_char_in  # cm^-3
            n_H = n_e
            t_c = te_sample[j] #* te_char_in  # K
            v_c = v_sample[j] #* v_char_in  # cm/s
            g_c1, g_c2 = func_g(t_c, fe21, fe24)

            phi_nu_c = func_phi_nu(nu_c, nu0_1354, t_c, v_c)
            I_1d[j] = phi_nu_c * n_e * n_H * g_c1

        # integration along the sampling line: the distance if defined by lines
        I_nu[k] = np.trapz(I_1d, lines_cm)

    # wv_arr -> nu_arr -> doppler_arr
    # doppler_arr = np.zeros(n_wv)
    # for k in range(n_wv):
    #    wv_c = wv_arr[k]
    #    nu_c = c_light/(wv_c*1.0e-8) # wavelength: 1 AA -> 1.0e-8 cm
    #    doppler_arr[k] = -(c_light/nu0_1354)*(nu_c - nu0_1354)
    # print(doppler_arr)
    doppler_arr = c_light * 1.0e-5 * (wv_arr - 1354.08) / 1354.08  # x 1.0e-5: cm/s to km/s

    return [I_nu, doppler_arr, wv_arr, lines_cm]

#%%
downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
# cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
# Define RAY
p1 = [-0.25, 0.48, 0.00]  # [-0.25, 0.25, -0.2499]
p2 = [0.25, 0.48, 0.00]  # [0.25, 0.75, 0.2499]
los_vec = np.subtract(np.array(p2), np.array(p1))
ray = subs_ds.ray(p1, p2)

nsample = ray[('gas', 'temperature')].value.size
#lines = np.linspace(0, np.linalg.norm(ray.start_point - ray.end_point), nsample)
linex = np.linspace(ray.start_point[0], ray.end_point[0], nsample)
liney = np.linspace(ray.start_point[1], ray.end_point[1], nsample)
linez = np.linspace(ray.start_point[2], ray.end_point[2], nsample)
lines = np.zeros(nsample)
for i in range(nsample):
    lines[i] = np.sqrt((linex[i] - ray.start_point[0]) ** 2
                       + (liney[i] - ray.start_point[1]) ** 2
                       + (linez[i] - ray.start_point[2]) ** 2)

# ray[('gas','velocity_x')].value, ray[('gas','velocity_y')].value, ray[('gas','velocity_z')].value

'''
Properly extract and rescale physical parameters from the model
'''

# Reproject velocity along the los vector:
#for i in range(ray.shape):
v_sample = ray[('gas', 'velocity_x')].value * los_vec[0] + ray[('gas', 'velocity_y')].value * los_vec[1] +\
                ray[('gas', 'velocity_z')].value * los_vec[2]

v_sample = np.divide(v_sample, np.linalg.norm(los_vec))
te_sample = ray[('gas', 'temperature')].value
rho_sample = ray[('gas', 'density')].value

#%%
[I_nu, doppler_arr, wv_arr, lines_cm] = func_intensity_fe21(lines, v_sample, te_sample, rho_sample,
                                                rho_char, te_char, v_char)


#%%
plt.ioff()
plt.rcParams.update({'font.size': 8})

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(lines_cm, v_sample, c='red')
axs[0, 0].set_title('LOS velocity profile')
axs[0, 1].plot(lines_cm, te_sample, c='tab:orange') #plot(x, y, 'tab:orange')
axs[0, 1].set_title('LOS Temperature profile')
axs[1, 0].plot(lines_cm, rho_sample, c='tab:green') #plot(x, -y, 'tab:green')
axs[1, 0].set_title('LOS density profile')
axs[1, 1].plot(wv_arr, I_nu, c='tab:blue') #plot(x, -y, 'tab:red')
axs[1, 1].set_title('Synthetic Fe XXI intensity')

plt.subplots_adjust(top=0.92,
                    bottom=0.085,
                    left=0.08,
                    right=0.975,
                    hspace=0.315,
                    wspace=0.145)


#plt.subplot_tool()
#plt.show()
# plt.plot(wv_arr, I_nu, c='red')
plt.savefig('./fe_xxi.png', dpi=250)
#plt.close()
