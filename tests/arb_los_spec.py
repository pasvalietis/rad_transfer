import os
os.environ["XUVTOP"] = "/home/ivan/Study/Astro/solar/xuv/CHIANTI_10.0.2_database"
from cProfile import label
from colorsys import yiq_to_rgb
import math
from turtle import color
import numpy as np
from scipy.optimize import curve_fit

import yt
from unyt import unyt_array
from unyt import cm as cm_, s as s_

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
    n_wv = 100  # set resolution of the wavelength
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
p1 = [-0.5, 0.48, 0.00]  # [-0.25, 0.25, -0.2499]
p2 = [0.5, 0.48, 0.00]  # [0.25, 0.75, 0.2499]
los_vec = np.subtract(np.array(p2), np.array(p1))
ray = subs_ds.ray(p1, p2)

def rot_ray(ds, p1, p2, phi):
    # rot_ray(subs_ds, p1, p2, 15*phi)
    """Rotate yt_ray object and return new ray and intersection points
    :param ds: Input dataset
    :param p1: First intersection point of the ray
    :param p2: Second intersection point of the ray
    :param phi: Rotation angle in degrees (astropy.units object)
    :return: [P1*, P2*, ray*]: New ray intersection points and yt ray object
    """
    # Case 1: 0 < phi < 90, rotation in X-Z plane

    p1_, p2_ = np.copy(p1), np.copy(p2)

    x_max, y_max, z_max = [ds.domain_width[i] for i in range(3)]  # In code_units
    phi_crit = (np.arctan(x_max / z_max)*u.rad).to(u.deg)   # Critical angle that is switching coordinate increment

    if phi < phi_crit:
        dz = ((x_max/2.) * np.tan(phi.to(u.rad))).value
        p1_[2] = p1[2] + dz
        p2_[2] = p2[2] - dz
    elif phi > phi_crit:
        dx = 0.5 * (x_max - z_max/(np.tan(phi.to(u.rad)))).value
        p1_[0] = p1[0] + dx
        p2_[0] = p2[0] - dx
    else:
        ValueError('Keep rotation angle in X-Z plane 0 < phi < 90')

    new_ray = ds.ray(p1_, p2_)

    return p1_, p2_, new_ray

#%%
def plot_params(phi, lines, wv_arr, v_sample, te_sample, rho_sample, I_nu):
    plt.ioff()
    plt.rcParams.update({'font.size': 6})

    fig, axs = plt.subplots(2, 2)

    v_sample.convert_to_units('km/s')
    axs[0, 0].plot(lines.to('Mm'), v_sample, c='red', linewidth=0.65)
    axs[0, 0].set_title('LOS velocity profile')
    axs[0, 0].set_xlabel('l.o.s. distance, Mm')
    axs[0, 0].set_ylabel('Velocity, km/s')
    f = lambda x: 1e8 * x / l_char
    g = lambda x: x * l_char / 1e8
    ax2 = axs[0, 0].secondary_xaxis("top", functions=(f, g))
    axs[0, 0].set_ylim(-150, 150)
    axs[0, 0].set_xlim(0, 300)

    axs[0, 1].semilogy(lines.to('Mm'), te_sample, c='tab:orange', linewidth=0.65)  # plot(x, y, 'tab:orange')
    axs[0, 1].set_title('LOS Temperature profile')
    axs[0, 1].set_xlabel('l.o.s. distance')
    axs[0, 1].set_ylabel('Temperature, K')
    axs[0, 1].set_ylim(1e6, 3e7)
    axs[0, 1].set_xlim(0, 300)
    ax2 = axs[0, 1].secondary_xaxis("top", functions=(f, g))

    axs[1, 0].semilogy(lines.to('Mm'), rho_sample, c='tab:green', linewidth=0.65)  # plot(x, -y, 'tab:green')
    axs[1, 0].set_title('LOS density profile')
    axs[1, 0].set_xlabel('l.o.s. distance')
    axs[1, 0].set_ylabel('Density, cm$^{-3}$')
    axs[1, 0].set_ylim(1e8, 2e9)
    axs[1, 0].set_xlim(0, 300)
    ax2 = axs[1, 0].secondary_xaxis("top", functions=(f, g))

    axs[1, 1].plot(wv_arr, I_nu, c='tab:blue', linewidth=0.65)  # plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Synthetic Fe XXI intensity')
    axs[1, 1].set_xlabel('Wavelength, $\AA$')
    axs[1, 1].set_ylabel('Intensity, erg cm$^{-2}$ s$^{-1}$')
    axs[1, 1].set_xlim(1351.5, 1356.5)
    axs[1, 1].set_ylim(0, 8e-12)

    plt.subplots_adjust(top=0.915,
                        bottom=0.083,
                        left=0.088,
                        right=0.981,
                        hspace=0.472,
                        wspace=0.208)

    # plt.subplot_tool()
    # plt.show()
    # plt.plot(wv_arr, I_nu, c='red')
    plt.savefig('./los_spectra/fe_xxi_phi_'+str(int(phi.value))+'.png', dpi=280)
    # plt.close()

# ray[('gas','velocity_x')].value, ray[('gas','velocity_y')].value, ray[('gas','velocity_z')].value

'''
Properly extract and rescale physical parameters from the model
'''
ang_res = 15
for i in range(ang_res):
    # Define new ray geometries
    phi = i * 90. / ang_res * u.deg  # Ray rotation angle

    new_ray = rot_ray(subs_ds, p1, p2, phi)[2]

    #%%
    nsample = new_ray[('gas', 'temperature')].value.size
    #lines = np.linspace(0, np.linalg.norm(ray.start_point - ray.end_point), nsample)
    linex = np.linspace(new_ray.start_point[0], new_ray.end_point[0], nsample)
    liney = np.linspace(new_ray.start_point[1], new_ray.end_point[1], nsample)
    linez = np.linspace(new_ray.start_point[2], new_ray.end_point[2], nsample)

    lines = unyt_array(np.zeros(nsample))*linex.units
    for i in range(nsample):
        lines[i] = np.sqrt((linex[i] - new_ray.start_point[0]) ** 2
                           + (liney[i] - new_ray.start_point[1]) ** 2
                           + (linez[i] - new_ray.start_point[2]) ** 2)

    # Reproject velocity along the los vector:
    #for i in range(ray.shape):
    v_sample = new_ray[('gas', 'velocity_x')] * los_vec[0] + new_ray[('gas', 'velocity_y')] * los_vec[1] +\
                    new_ray[('gas', 'velocity_z')] * los_vec[2]

    v_sample = np.divide(v_sample, np.linalg.norm(los_vec))
    #v_sample *= cm_/s_
    #v_sample = v_sample.convert_to_units('km/s')
    te_sample = new_ray[('gas', 'temperature')]#.value
    rho_sample = new_ray[('gas', 'density')]#.value


    #%%
    [I_nu, doppler_arr, wv_arr, lines_cm] = func_intensity_fe21(lines, v_sample.value, te_sample.value, rho_sample.value,
                                                    rho_char, te_char, v_char)

    plot_params(phi, lines, wv_arr, v_sample, te_sample, rho_sample, I_nu)




#%%

