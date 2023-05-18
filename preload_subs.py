import yt
import numpy as np
from yt.fields.particle_fields import obtain_relative_velocity_vector

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

def _subs_mass(field, data):
    renorm = 4.70958407245e+35
    return (renorm * data["gas", "mass"])

def load_subs(ds):
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

    ds.add_field(
        ("gas", "subs_mass"),
        function=_subs_mass,
        sampling_type="cell",
        units="g",
    )

    return ds
