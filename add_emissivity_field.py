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
from yt.fields.particle_fields import obtain_relative_velocity_vector
from yt.fields.vector_operations import get_bulk

#%%

class ThermalBremsstrahlungModel:
    def __init__(self, temperature_field, density_field, mass_field):
        self.temperature_field = temperature_field
        self.density_field = density_field
        self.mass_field = mass_field
        pass

    def setup_model(self, data_source):
        if isinstance(data_source, Dataset):
            ds = data_source
        else:
            ds = data_source.ds
        self.temperature_field = ds._get_field_info(self.temperature_field).name
        self.density_field = ds._get_field_info(self.density_field).name
        self.mass_field = ds._get_field_info(self.mass_field).name
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
        phot_field = photon_energy.to(u.erg).value * np.ones_like(chunk[self.mass_field].d)
        orig_shape = chunk[self.temperature_field].shape
        num_cells = len(chunk[self.mass_field])
        dens = chunk[self.density_field].d
        mass = chunk[self.density_field].d
        temp = chunk[self.temperature_field].d
        # em_data = dens**2.
        norm_energy = phot_field/(temp*kboltz)
        idV = dens / mass
        #gaunt = 1.5 #np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        #np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        #gaunt *= (np.sqrt(3.) / np.pi)

        gf = np.exp(0.5 * norm_energy) * k0(0.5 * norm_energy)
        gf *= (np.sqrt(3.) / np.pi)

        gaunt = 1 #.5 #np.nan_to_num(gf, nan=1.5) #- 7.01*np.ones_like(gf)

        # brm_49 emission field
        #em_data = (1e8 / 9.26) * np.exp(-norm_energy) / phot_field / np.sqrt(temp)
        # Aschwanden emissivity
        factor = 5.44436678165399e-39
        au_dist = 14959787070000.0 # One astronomical unit
        em_data = idV*factor*((dens**2.)/np.sqrt(np.abs(temp)))*np.exp(-np.abs(norm_energy))
        #em_data *= gaunt
        em_data *= 1./(phot_field) # to get flux in photons
        em_data *= 1./(hplanck) # to get flux in photons/(s cm^2 keV)
        em_data *= 1./(au_dist**2.)

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

indstype = 'subs'
path_test = "datacubes/id0/Blast.0020.vtk"
path_default = "datacubes/flarecs-id.0035.vtk"
path_subs = "datacubes/flarecs-id.0035_ss3.h5"
#path_subs = "datacubes/flarecs-id.0035_ss2.h5"

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    # "density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

if indstype == 'full':
    path = path_default
    ds = yt.load(path, units_override=units_override, default_species_fields='ionized')
elif indstype == 'subs':
    path = path_subs
    ds = yt.load(path)

# Add temperature field to a subsampled dataset // Talk about it to Sabastian
#%%

axis_names = ('x', 'y', 'z')
axis_order = ('x', 'y', 'z')

def _subs_density(field, data):
    renorm = 229730894.015
    return (data["grid", "density"] * renorm).in_units("g/cm**3")

def eint_from_etot(data):                                                                                                                                                                           
            eint = (                                                                                                                                                                                        
                data["grid", "total_energy"] - data["gas", "kinetic_energy_density"]
            )                                                                                                                                                                                               
            #if ("athena", "cell_centered_B_x") in self.field_list:
            eint -= data["gas", "magnetic_energy_density"]
            return eint

def _subs_velocity_field(comp):
    def _velocity(field, data):
        renorm = 1.#3.63014233846e+16
        return renorm*data["grid", f"momentum_{comp}"] / data["gas", "dens"]
    return _velocity

def _subs_specific_thermal_energy(field, data):
    return eint_from_etot(data) / data["grid", "density"]

def _subs_kinetic_energy_density(field, data):
    v = obtain_relative_velocity_vector(data)
    return 0.5 * data["gas", "dens"] * (v**2).sum(axis=0)

def _subs_magnetic_field_strength(field, data):
    xm = f"cell_centered_B_{axis_names[0]}"
    ym = f"cell_centered_B_{axis_names[1]}"
    zm = f"cell_centered_B_{axis_names[2]}"
    B2 = (data["grid", xm]) ** 2 + (data["grid", ym]) ** 2 + (data["grid", zm]) ** 2
    return np.sqrt(B2)

def _subs_magnetic_energy_density(field, data):
    B = data["gas", "magnetic_field_strength"]
    return 0.5 * B * B / (4.0 * np.pi) # mag_factors(B.units.dimensions)

def _subs_pressure(field, data):
    """M{(Gamma-1.0)*rho*E}"""
    gamma = 5./3.
    tr = (gamma - 1.0) * (
            data["gas", "dens"] * data["gas", "specific_thermal_energy"]
    )
    return tr

def _subs_temperature(field, data):
    pc = data.ds.units.physical_constants
    renorm = 1.046449052e+16#1e24
    mu = 0.5924489101195808
    return (mu * renorm * data["gas", "pressure"] / data["gas", "dens"] * pc.mh / pc.kboltz).in_units("K")

#%%


#%%
if path == path_subs:
    ds.add_field(
        ("gas", "dens"),
        function=_subs_density,
        sampling_type="cell",
        units="g/cm**3",
    )

    # ds.add_field

    for comp in "xyz":
        ds.add_field(
            ("gas", f"velocity_{comp}"),
            sampling_type="cell",
            function=_subs_velocity_field(comp),
            units='cm/s',
        )

    ds.add_field(
        ("gas", "magnetic_field_strength"),
        function=_subs_magnetic_field_strength,
        sampling_type="cell",
        units="G",
    )

    ds.add_field(
        ("gas", "magnetic_energy_density"),
        function=_subs_magnetic_energy_density,
        sampling_type="cell",
        units="dyn/cm**2",
    )

    ds.add_field(
        ("gas", "kinetic_energy_density"),
        function=_subs_kinetic_energy_density,
        sampling_type="cell",
        units="dyn/cm**2",
    )

    ds.add_field(
        ("gas", "specific_thermal_energy"),
        function=_subs_specific_thermal_energy,
        sampling_type="cell",
        units="erg/g",
    )

    ds.add_field(
        ("gas", "pressure"),
        function=_subs_pressure,
        sampling_type="cell",
        units="dyn/cm**2",
    )

    ds.add_field(
        ("gas", "temperature"),
        function=_subs_temperature,
        sampling_type="cell",
        units="K",
    )

#%%
# Plot a histogram of a field phys. parameter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# n_bins = 1000
ftype = 'gas' # specific field_type
fname = 'temperature'
bins = 10**(np.linspace(np.log10(ds.all_data()[(ftype, fname)].min()/10.),
            np.log10(ds.all_data()[(ftype, fname)].max()*10.),
            500))
count, bins, ignored = plt.hist(ds.all_data()[(ftype, fname)], density=True, log=True, bins=bins)

if fname == 'velocity_x':
    bins = 10 ** (np.linspace(np.log10(np.abs(ds.all_data()[(ftype, fname)]).min() / 10.),
                              np.log10(np.abs(ds.all_data()[(ftype, fname)]).max() * 10.),
                              500))
    count, bins, ignored = plt.hist(np.abs(ds.all_data()[(ftype, fname)]), density=True, log=True, bins=bins)
    #plt.xlim(np.abs(ds.all_data()[(ftype, fname)].min()) / 1e8, np.abs(ds.all_data()[(ftype, fname)].max()) * 10.)
else:
    plt.xlim(ds.all_data()[(ftype, fname)].min()/10., ds.all_data()[(ftype, fname)].max()*10.)
#plt.ylim(1e-7, 1e2)


plt.xscale('log')
plt.yscale('log')
#plt.title('Number density')
#plt.show()
#imgpath = 'img/phys_param_distributions/original/'
imgpath = 'img/phys_param_distributions/subsampled/'
plt.savefig(imgpath + 'resampled_'+fname+'_dist.eps')

#%%
if path == path_subs:
    thermal_model = ThermalBremsstrahlungModel("temperature", "dens", "mass")
else:
    thermal_model = ThermalBremsstrahlungModel("temperature", "density", "mass")

thermal_model.make_intensity_fields(ds)
#print(ds.all_data()['gas', 'xray_intensity_keV'])

'''
To be described later
'''

# Make a projection of intensity field

#%%
N = 512
norm_vec = [0.0, 0.0, 1.0]
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

#%%
fig, ax = plt.subplots()
data_img = np.array(prji)
imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale

pcm = ax.pcolor(X, Y, imag,
                        #norm=colors.LogNorm(vmin=1e-40, vmax=1e-23),
                        vmin=1e-24,
                        vmax=8e-21,
                        cmap='inferno', shading='auto')
int_units = str(prji.units)
fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
ax.set_xlabel('x, Mm')
ax.set_ylabel('y, Mm')

#plt.show()
figpath = './img/rad_tr_thermal_brem/'
plt.savefig(figpath + 'therm_brem_front_view_'+indstype+'.png')
#%%
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
