#
# Purpose:
#   This script is used to obtain the velocity, temperature, density and Fe XXI line
#   along different LOSs.
# Author:
#   Chengcai
# Update:
#   2021-10-26
#   2021-10-29
#   2022-01-27
#   2022-02-22
#   2022-02-24
#   2022-02-25
#%%
import os
os.environ["XUVTOP"] = "/home/ivan/Study/Astro/solar/xuv/chianti"
from cProfile import label
from colorsys import yiq_to_rgb
import math
from turtle import color
import numpy as np
from scipy.optimize import curve_fit

import pickle
import gc
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Athena_Reader
import athena_vtk_reader
import mhdpost_func

# ChiantiPy
import ChiantiPy.core as ch


# Astropy
from astropy import units as u
from astropy import constants as const
from astropy.modeling import models, fitting

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
# Example of thermal broadening at T=7.05
#
Te0 = 10.0**7.05
v_thermal = np.sqrt(2.0 * const.k_B * Te0 * u.K / (56.0* const.m_p))
v_thermal.to('m/s')
print(v_thermal.to('m/s'))

v_thermal = np.sqrt(2.0 * Kb * Te0/mfe)
print(v_thermal,"(cm/s)", const.k_B)

#dwv_o_wv = 1./(3.0e8)*math.sqrt((2.0*1.38e-23*1.0e7)/(56.0*1.66e-27))
#print(dwv_o_wv, dwv_o_wv*1354.08)

# 
# function to get g(tc) for both Fe21 and Fe24 lines
#
def func_g(tc, fe21, fe24):
    gt_1354 = np.interp(tc, fe21.Gofnt['temperature'], fe21.Gofnt['gofnt'])
    gt_192  = np.interp(tc, fe24.Gofnt['temperature'], fe24.Gofnt['gofnt'])
    return gt_1354, gt_192

#
# function: phi_nu
# 
# !!! notice: CGS unit in this function !!!
def func_phi_nu(nu, nu0, tc, vs):
    # thermal speed and boradening
    v_thermal = np.sqrt(2.0 * Kb * tc/mfe)
    delt_nud = (nu/c_light)*v_thermal
    
    # frequency shift
    delt_nu = nu - nu0
    
    # Relative velocity of iron
    v_ratio = vs/c_light
    
    part_head = 1./(math.sqrt(math.pi)*delt_nud)
    part_exp_in = -((delt_nu + nu0*v_ratio)/delt_nud)**2
    
    # phi_nu
    phi_nu = part_head*np.exp(part_exp_in)
    
    return phi_nu

#
# func: interpolate sampling lines
#
def func_interp_sample(ps, pe, nsample):
    lines = np.zeros(nsample)

    linex = np.linspace(ps[0], pe[0], nsample)
    liney = np.linspace(ps[1], pe[1], nsample)
    linez = np.linspace(ps[2], pe[2], nsample)

    for i in range(nsample):
        lines[i] = np.sqrt((linex[i] - ps[0])**2
                         + (liney[i] - ps[1])**2
                         + (linez[i] - ps[2])**2)

    points_sample = np.zeros((3, nsample))
    for ipt in range(nsample):
        points_sample[0, ipt] = linez[ipt]
        points_sample[1, ipt] = liney[ipt]
        points_sample[2, ipt] = linex[ipt]
    te_sample = mhdpost_func.func_interpol_3d(x, y, z, te3, points_sample, nsample)
    rho_sample = mhdpost_func.func_interpol_3d(x, y, z, rho3, points_sample, nsample)
    
    vz_sample = mhdpost_func.func_interpol_3d(x, y, z, vz3, points_sample, nsample)
    vy_sample = mhdpost_func.func_interpol_3d(x, y, z, vy3, points_sample, nsample)
    vx_sample = mhdpost_func.func_interpol_3d(x, y, z, vx3, points_sample, nsample)
    v_scale = np.sqrt(vx_sample**2 + vy_sample**2 + vz_sample**2)

    los_x = pe[0] - ps[0]
    los_y = pe[1] - ps[1]
    los_z = pe[2] - ps[2]
    los_scale = np.sqrt(los_x**2 + los_y**2 + los_z**2)
    v_sample = (vx_sample*los_x + vy_sample*los_y + vz_sample*los_z)/(los_scale)

    return [linex, liney, linez, lines, v_sample, te_sample, rho_sample]

#
# func: intensity_fe21
#
def func_intensity_fe21(lines, v_sample, te_sample, rho_sample, rho_char_in, te_char_in, v_char_in):
    #
    # (old 4.1 Fe XX1 all)
    #
    n_wv = 201 # set resolution of the wavelength
    wv_arr = np.linspace(1352, 1356, n_wv)
    I_nu = np.zeros(n_wv)
    lines_cm = lines*l_char

    for k in range(n_wv):
        wv_c = wv_arr[k]
        nu_c = c_light/(wv_c*1.0e-8) # wavelength: 1 AA -> 1.0e-8 cm
        
        I_1d = np.zeros(nsample)
        for j in range(nsample):
            n_e = rho_sample[j]*rho_char_in # cm^-3
            n_H = n_e
            t_c = te_sample[j]*te_char_in # K
            v_c = v_sample[j]*v_char_in # cm/s
            g_c1, g_c2 = func_g(t_c, fe21, fe24)
            
            phi_nu_c = func_phi_nu(nu_c, nu0_1354, t_c, v_c)
            I_1d[j] = phi_nu_c*n_e*n_H*g_c1
            
        # integration along the sampling line: the distance if defined by lines
        I_nu[k] = np.trapz(I_1d, lines_cm)

    # wv_arr -> nu_arr -> doppler_arr
    #doppler_arr = np.zeros(n_wv)
    #for k in range(n_wv):
    #    wv_c = wv_arr[k]
    #    nu_c = c_light/(wv_c*1.0e-8) # wavelength: 1 AA -> 1.0e-8 cm
    #    doppler_arr[k] = -(c_light/nu0_1354)*(nu_c - nu0_1354)
    #print(doppler_arr)
    doppler_arr = c_light * 1.0e-5 * (wv_arr - 1354.08) / 1354.08 # x 1.0e-5: cm/s to km/s

    return [I_nu, doppler_arr, wv_arr, lines_cm]


def fcun_doppler_width(wv_arr, doppler_arr, I_nu):
    # estimate gaussian parameters
    I_peak = np.amax(I_nu)
    ipeak = np.argmax(I_nu)
    res = np.where(I_nu >= 0.5*I_peak)
    iis = res[0][0] - 1
    iie = res[0][-1]
    wv_fwhm = np.fabs(wv_arr[iie] - wv_arr[iis])
    wv_peak = wv_arr[ipeak]
    v_peak = doppler_arr[ipeak]
    # to doppler width: fwhm -> doppler width
    #ratio = np.sqrt(2)/2.3548200450309493

    # fitting data us a Gaussian1D model
    g_init = models.Gaussian1D(amplitude=np.amax(I_nu), mean=wv_peak, stddev=wv_fwhm/2.3548200450309493)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, wv_arr, I_nu)
    
    # get doppler width and profile center
    wv_peak = g.mean.value
    wv_width = np.sqrt(2)*g.stddev.value

    v_peak = c_light * 1.0e-5 * (wv_peak - 1354.08) / 1354.08 #cm/s -> km/s
    v_width = c_light * 1.0e-5 * (wv_width) / 1354.08 #cm/s -> km/s

    return wv_peak, v_peak, wv_width, v_width


# ==============================================================================
# Main
# ==============================================================================
# ------------------------------------------------------------------------------
# 0. Contribution function g(T)
# ------------------------------------------------------------------------------
#%%
# Fe XXIV (192.04 A; sensitive to ~17 MK plasma -- used by Ryan's paper with Hinode/EIS)
# and/or Fe XXI (1354.08 A; sensitive to ~10 MK plasma -- measured by IRIS)
t = 10.**(6.0 + 0.01*np.arange(151.))
#%%
#fe21 = ch.ion('fe_21', temperature=t, eDensity=1.e+9, em=1.e+27)
#fe24 = ch.ion('fe_24', temperature=t, eDensity=1.e+9, em=1.e+27)
#%%
#fe21.gofnt(wvlRange=[1354., 1355.0],top=3, plot=False)
#%%
#fe24.gofnt(wvlRange=[192., 192.2],top=3, plot=False)
#%%
# restore from the saved pickle files running the above commands
#[fe21, fe24] = pickle.load(open("./data_chianti_fe/fe21_24.p", "rb"))
#%%
'''
f = open('fe21_24.p', 'wb') # Pickle file is newly created where foo1.py is
pickle.dump([fe21, fe24], f) # dump data to f
f.close()
'''
#%%
[fe21n, fe24n] = pickle.load(open("./data_chianti_fe/fe21_24.p", "rb"))
#%%
# ------------------------------------------------------------------------------
# 0.1 Initialize parameters
# ------------------------------------------------------------------------------
filename = 'flarecs-id.0035.vtk'
yc_chosen = 0.48

# ------------------------------------------------------------------------------
# 1. Read 3D vtk data
# ------------------------------------------------------------------------------
#%%
vtkfile = './datacubes/' + filename
var = athena_vtk_reader.read(vtkfile, outtype='cons', nscalars=0)
#%%
x = var['x']
y = var['y']
z = var['z']
time = var['time']
bx3 = var['bx']
by3 = var['by']
bz3 = var['bz']
rho3 = var['rho']
vx3 = var['mx']/rho3
vy3 = var['my']/rho3
vz3 = var['mz']/rho3
ek = 0.5*rho3*(vx3**2 + vy3**2 + vz3**2)
eb = 0.5*(bx3**2 + by3**2 + bz3**2)
gamma = 5./3.
p3 = (var['e'] - ek - eb)*(gamma - 1.0)
te3 = p3/rho3
del ek, eb, p3, bx3, by3, bz3
gc.collect()
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

#%%
# -----------------------------------
# 2. Plot lines with different te_scales
# -----------------------------------
nsample = 2001
str_sample = ['y-', 'x-', 'z-', 'xz-']
ps_list = [[0, 0.02, 0], [-0.25, yc_chosen, 0], [0, yc_chosen, -0.25], [-0.1, yc_chosen, -0.25]]
pe_list = [[0, 1.00, 0], [0.25, yc_chosen, 0],  [0, yc_chosen, 0.25],  [+0.1, yc_chosen, 0.25]]

for isample in range(4):
    ps = ps_list[isample]
    pe = pe_list[isample]
    [linex, liney, linez, lines, v_sample, te_sample, rho_sample] = func_interp_sample(ps, pe, nsample)

    fig = plt.figure(figsize=(10,10))
    # ax1: te profile
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('(a) Temperature along LOS (K)')
    ax1.set_xlabel('LOS distance (L$_0$)')
    ax1.text(0.05, 0.95, 'Time={0:.1f}t$_0$'.format(time), transform=ax1.transAxes, color='black')
    t = ax1.text(0.05, 0.8, 'LOS along '+ str_sample[isample]
        + ',\n([{0:.2f},{1:.2f},{2:.2f}] ~ [{3:.2f},{4:.2f},{5:.2f}])'.format(ps[0],ps[1],ps[2],pe[0],pe[1],pe[2]),
        transform=ax1.transAxes, color='black')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

    # ax2: rho profile
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('(b) Density along LOS (cm$^{-3}$)')
    ax2.set_xlabel('LOS distance (L$_0$)')

    # ax3: v profile
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title('(c) Velocity Component along LOS (km/s)')
    ax3.set_xlabel('LOS distance (L$_0$)')

    # ax4: Fe XXI profile
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_xlabel('Wavelength')
    ax4.axvspan(1352, 1354.08, facecolor='gray', alpha=0.1)
    ax4.axvspan(1354.08, 1356, facecolor='red', alpha=0.1)
    ax41 = ax4.twiny()
    ax41.text(0.05, 0.95, "(d) Synthetic Intensity (Fe XXI 1354$\AA$)", transform=ax41.transAxes, color='black')
    ax41.set_xlabel("Wave Length $(\AA)$")
    ax41.set_xlabel("Doppler Velocity (km/s)")

    # enter the te_scale loop
    colors = ['b', 'g', 'orange']
    for ite in range(3):
        te_char_c = te_char*te_scale_list[ite]
        v_char_c = v_char*np.sqrt(te_scale_list[ite])
        [I_nu, doppler_arr, wv_arr, lines_cm] = func_intensity_fe21(lines, v_sample, te_sample, rho_sample,
                                                rho_char, te_char_c, v_char_c)
        # try v==0, pure thermal broadening
        if (ite == 1):
            # get mean density and rho^2 weighted temperature
            rho_mean = np.mean(rho_sample)
            rho_mean_arr = np.ones(nsample) * rho_mean
            rho2_wt_te = 0.9e7/te_char_c
            rho2_wt_te_arr = np.ones(nsample) * rho2_wt_te
            [I_nu_v0, doppler_arr_v0, wv_arr_v0, lines_cm_v0] = func_intensity_fe21(lines, v_sample*0.0, rho2_wt_te_arr, rho_mean_arr,
                                                rho_char, te_char_c, v_char_c)

        # plot out
        ax1.plot(lines, te_sample*te_char_c, c=colors[ite], label='T$_0$ Scale={0:.1f}'.format(te_scale_list[ite]))
        ax2.plot(lines, rho_sample*rho_char, c='purple')
        ax3.plot(lines, v_sample*v_char_c*1.0e-5, c=colors[ite], label='T$_0$ Scale={0:.1f}'.format(te_scale_list[ite]))
        ax4.plot(wv_arr, I_nu, c=colors[ite])
        if (ite ==1):
            ax4.plot(wv_arr_v0, I_nu_v0, c=colors[ite], ls='--')
            
            # width
            wv_peak, v_peak, fwhm, del_v = fcun_doppler_width(wv_arr, doppler_arr, I_nu)
            wv_peak_v0, v_peak_v0, fwhm_v0, del_v0 = fcun_doppler_width(wv_arr_v0, doppler_arr_v0, I_nu_v0)
            t = ax4.text(0.05, 0.8, 'V$_c$={0:.1f} km/s,\nw={1:.2f}$\AA$ ({2:.1f}km/s)'.format(v_peak, fwhm, del_v), 
                transform=ax4.transAxes, color=colors[ite])
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

            t2 = ax4.text(0.05, 0.65, 'Dashed: T={0:.1f}MK,\nw$_0$={1:.2f}$\AA$ ({2:.1f}km/s)'.format(rho2_wt_te*te_char_c*1.0e-6, 
                fwhm_v0, del_v0),
                transform=ax4.transAxes, color=colors[ite])
            t2.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

        ax41.set_xlim([doppler_arr[0], doppler_arr[-1]])
    ax1.legend()
    ax3.legend()
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, right=0.9, top=0.95, bottom=0.05)
    plt.savefig('./fig_1los_'+str_sample[isample]+'_time{0:.2f}.pdf'.format(time), dpi=300)

#%%
# -------------------------------
#  3. Scan at different z for LOS x-
# -------------------------------
'''
nwv = len(wv_arr)
nz = len(z)
I_nu_arr = np.zeros((nwv, nz))
I_nu_arr_v0 = np.zeros((nwv, nz))

v_peak_arr = np.zeros(nz)
dev_v_arr = np.zeros(nz)
fwhm_arr = np.zeros(nz)

v_peak_arr_v0 = np.zeros(nz)
dev_v_arr_v0 = np.zeros(nz)
fwhm_arr_v0 = np.zeros(nz)

for k in range(nz):
    zc = z[k]
    ps = [-0.25, yc_chosen, zc]
    pe = [ 0.25, yc_chosen, zc]
    [linex, liney, linez, lines, v_sample, te_sample, rho_sample] = func_interp_sample(ps, pe, nsample)
    [I_nu, doppler_arr, wv_arr, lines_cm] = func_intensity_fe21(y, lines, v_sample, te_sample, rho_sample,
                                                rho_char, te_char, v_char)
    [I_nu_v0, doppler_arr_v0, wv_arr_v0, lines_cm_v0] = func_intensity_fe21(y, lines, v_sample*0.0, np.zeros(nsample)+10.0**7.05/te_char_c, rho_sample,
                                                rho_char, te_char, v_char)
    
    wv_peak, v_peak, fwhm, del_v = fcun_doppler_width(wv_arr, doppler_arr, I_nu)
    wv_peak_v0, v_peak_v0, fwhm_v0, del_v0 = fcun_doppler_width(wv_arr_v0, doppler_arr_v0, I_nu_v0)
   
    I_nu_arr[0:nwv, k] = I_nu[0:nwv]
    I_nu_arr_v0[0:nwv, k] = I_nu_v0[0:nwv]
    v_peak_arr[k] = v_peak
    v_peak_arr_v0[k] = v_peak_v0
    dev_v_arr[k] = del_v
    dev_v_arr_v0[k] = del_v0
    fwhm_arr[k] = fwhm
    fwhm_arr_v0[k] = fwhm_v0

# Plot 2D 
fig = plt.figure(figsize=(10, 10))
varmax = np.amax(I_nu_arr_v0)
varmin = 0
# ax1: I_nu_2d
ax1 = fig.add_subplot(2,2,1)
ax1.pcolormesh(z, wv_arr, I_nu_arr, vmax=varmax, vmin=varmin, cmap='jet', rasterized=True)
ax1.plot([z[0], z[-1]], [1354.08, 1354.08], c='gray')
ax1.set_xticklabels([])
ax1.set_ylabel('Wave Length')
ax1.set_title('(a) Fe XXI Profile at y={0:.2f}L$_0$'.format(yc_chosen))
ax1.text(0.05, 0.95, 'LOS along x- direction', transform=ax1.transAxes, color='white')

# ax2: I_nu_2d without velocity
ax2 = fig.add_subplot(2,2,2)
ax2.pcolormesh(z, wv_arr, I_nu_arr_v0, vmax=varmax, vmin=varmin, cmap='jet', rasterized=True)
ax2.plot([z[0], z[-1]], [1354.08, 1354.08], c='gray')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_title('(b) Same as the left but V==0')
# Colorbar
#axins1 = inset_axes(ax2,
#            width="80%",  # width = 50% of parent_bbox width
#            height="5%",  # height : 5%
#            loc='lower center')
#fig.colorbar(im, cax=axins1, orientation="horizontal", ticks=[varmin, varmax])
#axins1.xaxis.set_ticks_position("top")

# ax3, ax4
ax3 = fig.add_subplot(2,2,3)
ax3.plot(z, v_peak_arr, label='V$_{center}$')
ax3.plot(z, v_peak_arr_v0, label='(Flow==0)')
ax3.plot(z, dev_v_arr, label='$\Delta$V')
ax3.plot(z, dev_v_arr_v0, label='$\Delta$V(Flow==0)')
ax3.set_xlabel('z (L$_0$)')
ax3.legend()

ax4 = fig.add_subplot(2,2,4)
ax4.plot(z, fwhm_arr, label='w$_{center}$')
ax4.plot(z, fwhm_arr_v0, label='(Flow==0)')
ax4.set_xlabel('z (L$_0$)')
ax4.legend()

plt.subplots_adjust(wspace=0.2, hspace=0.05, left=0.1, right=0.9, top=0.95, bottom=0.05)
plt.savefig('./fig_lineprofile_2d_x-time{0:.2f}.pdf'.format(time), dpi=300)
'''