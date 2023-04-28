import yt
yt.enable_parallelism()
import numpy as np
import pickle
from pathlib import Path
from yt.data_objects.static_output import Dataset
import math

from scipy.special import k0
from astropy import constants as const
from astropy import units as u
#%%

class ThermalBremsstrahlungModel:
    def __init__(self, temperature_field, density_field, cell_mass):
        self.temperature_field = temperature_field
        self.density_field = density_field
        self.cell_mass = cell_mass
        pass

    def setup_model(self, data_source):
        if isinstance(data_source, Dataset):
            ds = data_source
        else:
            ds = data_source.ds
        self.temperature_field = ds._get_field_info(self.temperature_field).name
        self.density_field = ds._get_field_info(self.density_field).name
        self.emission_field = ds._get_field_info(self.cell_mass).name
        self.ftype = self.temperature_field[0]

    # def do_something_with(data):
    #     data = np.zeros(10000000)
    #     output = np.zeros_like(data)
    #     chunk_size = 1024
    #     for i_chunk in range(int(ceil(len(data) / chunk_size))):
    #         data_chunk = data[i_chunk * chunk_size:(i_chunk + 1) * chunk_size]
    #         output_chunk = process_data(data_chunk)
    #         output[i_chunk * chunk_size:(i_chunk + 1) * chunk_size] = output_chunk
    #     return output

    def process_data(self, chunk):

        pc = ds.units.physical_constants

        kboltz = 1.3807e-16  # Boltzmann's constant
        hplanck = 6.6261e-27  # Planck's constant cgs

        photon_energy = 6. * u.keV
        photon_energy_erg = photon_energy.to(u.erg)
        phot_field = photon_energy.to(u.erg).value * np.ones_like(chunk[self.emission_field].d)
        orig_shape = chunk[self.temperature_field].shape
        num_cells = len(chunk[self.emission_field])
        dens = chunk[self.density_field].d
        temp = chunk[self.temperature_field].d
        # em_data = dens**2.
        norm_energy = phot_field/(temp*kboltz)
        #gaunt = 1.5 #np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        #np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        #gaunt *= (np.sqrt(3.) / np.pi)

        gf = np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        gf *= (np.sqrt(3.) / np.pi)

        gaunt = 1.5 #np.nan_to_num(gf, nan=1.5) #- 7.01*np.ones_like(gf)

        # brm_49 emission field
        #em_data = (1e8 / 9.26) * np.exp(-norm_energy) / phot_field / np.sqrt(temp)
        # Aschwanden emissivity
        factor = 5.44436678165399e-39
        em_data = factor*((dens**2.)/np.sqrt(np.abs(temp)))*np.exp(-np.abs(norm_energy))
        em_data *= gaunt
        em_data *= 1./(hplanck) # to get photon flux in

        #print('num cells', num_cells)
        #ncells = 0
        print(len(em_data))
        return em_data

    def make_intensity_fields(
        self,
        ds,
    ):
        self.setup_model(ds)
        ftype = self.ftype
        ei_name = (ftype, f"xray_intensity_keV")
        ei_dname = rf"I_{{X}} (keV)"

        def _intensity_field(field, data):
            ret = data.ds.arr(
                self.process_data(data),
                "photons/cm**3/s/arcsec**2",
            )
            return ret

        ds.add_field(
            ei_name,
            function=_intensity_field,
            display_name=ei_dname,
            sampling_type="local",
            units="photons/cm**3/s/arcsec**2",
        )

path_test = "datacubes/id0/Blast.0020.vtk"
path_default = "datacubes/flarecs-id.0035.vtk"
path_subs = "datacubes/flarecs-id.0035_ss3.h5"
path = path_default

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    # "density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

# Add temperature field to a subsampled dataset // Talk about it to Sabastian
#%%
def _temperature(field, data):
    pc = data.ds.units.physical_constants
    return (data.ds.mu * data["gas", "pressure"] / data["gas", "density"] * pc.mh / pc.kboltz).in_units("K")

def _subs_density(field, data):
    return (data["grid", "density"] * 229730894.015).in_units("g/cm**3")

def _subs_eint_from_etot(data):
    return _subs_eint_from_etot(data) / data["gas", "dens"]

def _subs_specific_thermal_energy(field, data):
    return _subs_eint_from_etot(data) / data["gas", "dens"]

def _subs_pressure(field, data):
    """M{(Gamma-1.0)*rho*E}"""
    ftype = 'grid'
    tr = ((5./3.) - 1.0) * (data[ftype, "dens"] * data[ftype, "specific_thermal_energy"])
    return tr

def _subs_temperature(field, data):
    pc = data.ds.units.physical_constants
    renorm = 1e24
    mu = 0.5924489101195808
    return (mu * renorm * data['grid', 'total_energy'] / data["gas", "dens"] * pc.mh / pc.kboltz).in_units("K")

#%%
ds = yt.load(path, units_override=units_override, default_species_fields='ionized')


#%%
if path == path_subs:
    ds.add_field(
        ("gas", "dens"),
        function=_subs_density,
        sampling_type="cell",
        units="g/cm**3",
    )

    ds.add_field(
        ("gas", "temperature"),
        function=_subs_temperature,
        sampling_type="cell",
        units="K",
    )


#%%
#%%
# Plot a histogram of a field phys. parameter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# n_bins = 1000
fname = 'temperature'
bins = 10**(np.linspace(np.log10(ds.all_data()[('gas', fname)].min()/10.),
            np.log10(ds.all_data()[('gas', fname)].max()*10.),
            500))
count, bins, ignored = plt.hist(ds.all_data()[('gas', fname)], density=True, log=True, bins=bins)
plt.xlim(ds.all_data()[('gas', fname)].min()/10., ds.all_data()[('gas', fname)].max()*10.)
plt.xscale('log')
plt.yscale('log')
#plt.title('Number density')
plt.show()
plt.savefig('resampled_temp_dist.png')
#%%
if path == path_subs:
    thermal_model = ThermalBremsstrahlungModel("temperature", "dens", "cell_mass")
else:
    thermal_model = ThermalBremsstrahlungModel("temperature", "density", "cell_mass")

thermal_model.make_intensity_fields(ds)
#print(ds.all_data()['gas', 'xray_intensity_keV'])

'''
To be described later
'''

# Make a projection of intensity field

#%%
N = 512
norm_vec = [1.0, 1.0, 1.0]
prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
                        [0.0, 0.5, 0.0],  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        'xray_intensity_keV',  # respective field that is being projected
                        north_vector=[0.0, 1.0, 0.0]
                        )

Mm_len = 1 # ds.length_unit.to('Mm').value

X, Y = np.mgrid[-0.5*150*Mm_len:0.5*150*Mm_len:complex(0, N),
       0*Mm_len:150*Mm_len:complex(0, N)]

fig, ax = plt.subplots()
data_img = np.array(prji)
imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale

pcm = ax.pcolor(X, Y, imag,
                        norm=colors.LogNorm(vmin=1e7, vmax=1e10),
                        #vmin=0.0,
                        #vmax=0.04,
                        cmap='inferno', shading='auto')
int_units = str(prji.units)
fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
ax.set_xlabel('x, Mm')
ax.set_ylabel('y, Mm')

#plt.show()
plt.savefig('full_imag_isometric.png')

# # Considering a downsampled dataset
# u = ds.units
# norm = 1. * u.dyn / u.cm**2
# renorm = norm.to('code_pressure')
# e0 = yt.YTQuantity(1.0, "K")
#
# def _temperature(field, data):
#     return (
#         e0 * data['grid', 'total_energy'] / renorm
#     ).in_units("K")
#
# ds.add_field(
#     ("gas", "temperature"),
#     function=_temperature,
#     sampling_type="cell",
#     units="K",
# )

   #%%
# def _emissivity_field(field, data):
#     ret = data.ds.arr(
#         self.process_data("emissivity_field", data, spectral_norm, fluxf=eif),
#         "keV/s",
#     )
#     idV = data[ftype, "density"] / data[ftype, "mass"]
#     I = dist_fac * ret * idV
#     return I.in_units("erg/cm**3/s/arcsec**2")
#
# ds.add_field(
#     ei_name,
#     function=_emissivity_field,
#     display_name=ei_dname,
#     sampling_type="local",
#     units="erg/cm**3/s/arcsec**2",
#     force_override=force_override,
# )
# print(ds.all_data()['gas', 'xray_intensity_keV'])
#ds.all_data()['gas', 'xray_intensity_keV']

#%%
