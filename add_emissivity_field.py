import yt
yt.enable_parallelism()
import numpy as np
from yt.data_objects.static_output import Dataset
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

    def process_data(self, chunk):
        norm = chunk[self.emission_field].d
        orig_shape = chunk[self.temperature_field].shape
        num_cells = len(chunk[self.emission_field])
        dens = chunk[self.density_field].d
        temp = chunk[self.temperature_field].d
        em_data = dens**2.
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
#path_subs = "datacubes/flarecs-id.0035_ss3.h5"

# L_0 = (1.5e8, "m")
# units_override = {
#     "length_unit": L_0,
#     "time_unit": (109.8, "s"),
#     "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
#     # "density_unit": (2.5e14, "kg/m**3"),
#     "velocity_unit": (1.366e6, "m/s"),
#     "temperature_unit": (1.13e8, "K"),
# }

'''
# Add temperature field to a subsampled dataset // Talk about it to Sabastian 

def _temperature(field, data):
    pc = data.ds.units.physical_constants
    return (data.ds.mu * data["gas", "pressure"] / data["gas", "density"] * pc.mh / pc.kboltz)
    
ds.add_field(
("gas", "temperature"),
function=_temperature,
sampling_type="cell",
units="K",
)
'''

ds = yt.load(path_test, default_species_fields='ionized') #, units_override=units_override) #, default_species_fields='ionized')

#%%
thermal_model = ThermalBremsstrahlungModel("temperature", "density", "cell_mass")

thermal_model.make_intensity_fields(ds)
print(ds.all_data()['gas', 'xray_intensity_keV'])

'''
To be described later
'''

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

