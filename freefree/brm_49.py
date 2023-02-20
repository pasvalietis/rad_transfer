import numpy as np 
from acgaunt import acgaunt


def brem_49(kt, E):
	"""
	The function calculates the optically thin continuum thermal bremmstrahlung
	photon flux incident on the Earth from an isothermal plasma on the Sun.
	Normalization is for an emission measure on the sun of 1.e49cm^-3
	Category Spectra
	
	Parameters
	----------
	E : energy vector in keV
	kt 	   : plasma temperature in keV
	Returns
	-------
	Differential photon flux in units of photons/(cm2 s keV) per (1e49 cm-3 emission measure)
	Examples
	--------
	>>> flux = brem_49([30], 0.21)
	Notes
	-----
	Calls acgaunt.py
	"""

	kt0 = kt if kt > 0.1 else 0

	norm_energy = E/kt0
	norm_energy[norm_energy >= 50] = 50

	result = (1e8 / 9.26) * np.concatenate((list(map(acgaunt, np.divide(12.3985, E),
		np.divide(kt0, 0.08617)*np.ones(len(E))))), axis=1)[0]\
		* np.exp(-norm_energy) / E / np.sqrt(kt0)

	return result

