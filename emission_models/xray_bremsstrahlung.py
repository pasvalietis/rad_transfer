import yt
import numpy as np
import pickle
from pathlib import Path
from yt.data_objects.static_output import Dataset
import math

from scipy.special import k0
from scipy.integrate import quad

from astropy import constants as const
from astropy import units as u
from yt.fields.particle_fields import obtain_relative_velocity_vector
from yt.fields.vector_operations import get_bulk

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
        self.left_edge = ds.domain_left_edge
        self.right_edge = ds.domain_right_edge
        self.L_0 = 1.5e10
        self.domain_dimensions = ds.domain_dimensions

    # def do_something_with(data):
    #     data = np.zeros(10000000)
    #     output = np.zeros_like(data)
    #     chunk_size = 1024
    #     for i_chunk in range(int(ceil(len(data) / chunk_size))):
    #         data_chunk = data[i_chunk * chunk_size:(i_chunk + 1) * chunk_size]
    #         output_chunk = process_data(data_chunk)
    #         output[i_chunk * chunk_size:(i_chunk + 1) * chunk_size] = output_chunk
    #     return output

    def process_data(self, chunk, spec_param):

        kboltz = 1.3807e-16  # Boltzmann's constant
        hplanck = 6.6261e-27  # Planck's constant cgs

        emin = spec_param["emin"]
        emax = spec_param["emax"]

        nbins = 100

        photon_energy = 6. * u.keV
        photon_energy_erg = photon_energy.to(u.erg)
        phot_field = photon_energy.to(u.erg).value * np.ones_like(chunk[self.mass_field].d)
        orig_shape = chunk[self.temperature_field].shape
        num_cells = len(chunk[self.mass_field])
        dens = chunk[self.density_field].d
        mass = chunk[self.mass_field].d
        temp = chunk[self.temperature_field].d
        # em_data = dens**2.
        norm_energy = phot_field / (temp*kboltz)
        sizes = np.abs(self.right_edge - self.left_edge)
        idV = mass / dens
        #for i in range(3):
        #idV *= ((self.L_0 / self.domain_dimensions[i]) * sizes[i])

        dA = 1.
        for i in range(2):
            dA *= ((self.L_0 / self.domain_dimensions[i]) * sizes[i])

        # idV = 4.74e17 # mass / dens
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
        #ORIGINAL
        #em_data = idV*factor*((dens**2.)/np.sqrt(np.abs(temp)))*np.exp(-np.abs(norm_energy))
        ##em_data *= gaunt
        #em_data *= 1./(phot_field) # to get flux in photons
        #em_data *= 1./(hplanck) # to get flux in photons/(s cm^2 keV)
        #em_data *= 1./(au_dist**2.)

        def integrand(dens, temp, energy):
            intd = 8.1e-39 * np.exp(-np.abs((energy*u.keV).to(u.erg).value / temp*kboltz))
            intd *= (dens * dens) / np.sqrt(temp)
            #intd *= np.exp(-energy)
            return intd

        def energy_int(dens, temp, emin, emax):
            return quad(lambda energy: integrand(dens, temp, energy), emin, emax)

        vect_eint = np.vectorize(energy_int)
        res = vect_eint(dens, temp, emin, emax)[0]
        em_data = res

        # TEST
        # em_data = 8.1e-39 * np.exp(-np.abs(norm_energy)) * ((dens**2.)/np.sqrt(np.abs(temp))) * dA #* idV
        # angular_factor = 1. / (4. * np.pi * ((u.rad).to(u.arcsec)) ** 2.)
        # em_data *= angular_factor / photon_energy_erg.value
        # print('num cells', num_cells)
        # ncells = 0
        return em_data

    def make_intensity_fields(self, ds, emin, emax):
        self.setup_model(ds)
        ftype = self.ftype
        ei_name = (ftype, f"xray_intensity_keV")
        ei_dname = rf"I_{{X}} (keV)"

        emin = emin
        emax = emax
        spec_param = {"emin": emin, "emax": emax}

        def _intensity_field(field, data):
            ret = data.ds.arr(
                self.process_data(data, spec_param=spec_param),
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