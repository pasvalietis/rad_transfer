import sunpy.map
from xrtpy.response.temperature_from_filter_ratio import temperature_from_filter_ratio as tffr
import matplotlib.pyplot as plt

def plt_trans(filt='Ti-poly'):
    import xrtpy
    import matplotlib.pyplot as plt

    channel = xrtpy.response.Channel(filt)

    plt.figure(figsize=(10,6))
    plt.plot(channel.wavelength, channel.transmission, label=f"{channel.name}")

    plt.title('X-Ray Telescope', fontsize=15)
    plt.xlabel('Wavelength [Angsrom]', fontsize=15)
    plt.ylabel('Transmittance', fontsize=15)

    plt.legend(fontsize=20)
    plt.xlim(-5, 80)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(color='lightgrey')
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Testing of temperature_from_filter_ratio (IDL xrt_teem.pro)

events_path = '/home/saber/Downloads/2013_lvl1/'

ev1 = 'L1_XRT20130515_044714.1.fits'
ev2 = 'L1_XRT20130515_045708.6.fits'

map1 = sunpy.map.Map(events_path + ev1)
map2 = sunpy.map.Map(events_path + ev2)

T_EM = tffr(map1, map2)
T_e = T_EM.Tmap

def plot_temp():
    import matplotlib.pyplot as plt
    import numpy as np
    from sunpy.coordinates.sun import B0, angular_radius
    from sunpy.map import Map

    # To avoid error messages from sunpy we add metadata to the header:
    rsun_ref = 6.95700e08
    hdr1 = map1.meta
    rsun_obs = angular_radius(hdr1["DATE_OBS"]).value
    dsun = rsun_ref / np.sin(rsun_obs * np.pi / 6.48e5)
    solarb0 = B0(hdr1["DATE_OBS"]).value
    hdr1["DSUN_OBS"] = dsun
    hdr1["RSUN_REF"] = rsun_ref
    hdr1["RSUN_OBS"] = rsun_obs
    hdr1["SOLAR_B0"] = solarb0

    fig = plt.figure()
    # We could create a plot simply by doing T_e.plot(), but here we choose to make a linear plot of T_e
    m = Map((10.0 ** T_e.data, T_e.meta))
    m.plot(title="Derived Temperature", vmin=2.0e6, vmax=1.2e7, cmap="turbo")
    m.draw_limb()
    m.draw_grid(linewidth=2)
    cb = plt.colorbar(label="T (K)")


plot_temp()
plt.show()

