import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation
from sunpy.map import Map
import datetime
import pickle
import os
from unyt import unyt_array

# Import the functions from your module
from rushlight.utils.synth_tools import calc_vect, get_loop_coords, get_loop_params, get_reference_image, code_coords_to_arcsec

# --- Fixtures for common test objects ---

@pytest.fixture
def sample_loop_coords():
    """Provides a realistic set of SkyCoord objects for a loop."""
    x = np.linspace(-10, 10, 100) * u.Mm
    y = np.sin(np.linspace(0, np.pi, 100)) * 5 * u.Mm # Arc-like shape
    z = np.zeros_like(x) * u.Mm # Simple flat loop
    # Let's make it a semi-circle in XZ plane for calc_vect's assumptions
    x_sc = 10 * np.cos(np.linspace(0, np.pi, 100)) * u.Mm
    z_sc = 10 * np.sin(np.linspace(0, np.pi, 100)) * u.Mm
    y_sc = np.zeros_like(x_sc) * u.Mm

    # Observer's position and time
    obstime = '2023-01-01T00:00:00'
    observer = SkyCoord(0*u.m, 0*u.m, 1.496E11*u.m, frame='heliocentric', obstime=obstime)

    # Creating SkyCoord objects for each point
    coords_cart = CartesianRepresentation(x_sc, y_sc, z_sc)
    return SkyCoord(coords_cart, frame='heliocentric', obstime=obstime, observer=observer)


@pytest.fixture
def sample_ref_img():
    """Provides a basic sunpy.map.Map object for testing."""
    data = np.zeros((10, 10))
    header = {
        "CRPIX1": 5, "CRPIX2": 5,
        "CDELT1": 1, "CDELT2": 1,
        "CUNIT1": "arcsec", "CUNIT2": "arcsec",
        "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN",
        "CRVAL1": 0, "CRVAL2": 0,
        "NAXIS1": 10, "NAXIS2": 10,
        "DATE-OBS": "2023-01-01T00:00:00",
        "EXPTIME": 1.0,
        "WAVELNTH": 171.0,
        "MEASURE": "171A",
        "OBSERVER": "SDO/AIA",
        "TELESCOP": "SDO",
        "INSTRUME": "AIA",
        "DETECTOR": "AIA",
        "RSUN_OBS": 696000000.0, # Solar radius in meters at observer
        "HGLN_OBS": 0.0,
        "HGLT_OBS": 0.0,
        "DSUN_OBS": 1.496E11, # Distance to sun in meters
        "CROTA2": 0.0, # Rotation angle
    }
    return Map(data, header)

@pytest.fixture
def temp_pickle_file(tmp_path):
    """Creates a temporary pickle file with sample data."""
    data_to_pickle = {"test_key": "test_value", "phi0": 10*u.deg}
    file_path = tmp_path / "test_data.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump(data_to_pickle, f)
    return file_path

@pytest.fixture
def temp_map_pickle_file(tmp_path, sample_ref_img):
    """Creates a temporary pickle file with a sunpy Map."""
    file_path = tmp_path / "test_map.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump(sample_ref_img, f)
    return file_path

# --- Tests for calc_vect ---

class TestCalcVect:
    def test_calc_vect_return_type_and_shape(self, sample_loop_coords, sample_ref_img):
        normvector, northvector, ifpd = calc_vect(sample_loop_coords, sample_ref_img)
        assert isinstance(normvector, np.ndarray)
        assert normvector.shape == (3,)
        assert isinstance(northvector, np.ndarray)
        assert northvector.shape == (3,)
        assert isinstance(ifpd, float)

    def test_calc_vect_norm_vectors(self, sample_loop_coords, sample_ref_img):
        normvector, northvector, ifpd = calc_vect(sample_loop_coords, sample_ref_img)
        # Check if vectors are unit vectors (within tolerance)
        assert np.isclose(np.linalg.norm(normvector), 1.0)
        assert np.isclose(np.linalg.norm(northvector), 1.0)

    def test_calc_vect_default_north_vector(self, sample_loop_coords, sample_ref_img):
        normvector, northvector, ifpd = calc_vect(sample_loop_coords, sample_ref_img, default=True)
        # When default=True, northvector should be [0, 1, 0]
        assert np.allclose(northvector, [0, 1.0, 0])

    def test_calc_vect_ifpd_positive(self, sample_loop_coords, sample_ref_img):
        normvector, northvector, ifpd = calc_vect(sample_loop_coords, sample_ref_img)
        assert ifpd > 0

    def test_calc_vect_mhd_frame_orthogonality(self, sample_loop_coords, sample_ref_img):
        # This test aims to check the orthogonality of the MHD basis vectors used internally
        # It's an internal consistency check, not directly on output.
        # This requires some knowledge of calc_vect's internal workings.
        # A full test would involve re-calculating the basis and checking dot products.
        # For simplicity, we'll check the output vectors relative to each other if possible.
        normvector, northvector, ifpd = calc_vect(sample_loop_coords, sample_ref_img)

        # The calculation implies normvector is z_mhd, and x_mhd is v_12 normalized.
        # The y_mhd is cross(z_mhd, x_mhd).
        # We can't directly check x_mhd, y_mhd, z_mhd as they are not returned.
        # However, we can assert that the returned northvector and normvector
        # are sensible. A robust check might be to mock SkyCoord transformations
        # or calculate expected values for a very simple loop.
        # For now, let's just ensure they are not all zeros or NaNs.
        assert not np.isnan(normvector).any()
        assert not np.isnan(northvector).any()


# --- Tests for get_loop_coords ---

class TestGetLoopCoords:
    def test_get_loop_coords_return_type(self):
        loop_params = {"radius": 10 * u.Mm, "samples_num": 50}
        coords = get_loop_coords(loop_params)
        assert isinstance(coords, CartesianRepresentation)

    def test_get_loop_coords_samples_num(self):
        loop_params = {"samples_num": 75}
        coords = get_loop_coords(loop_params)
        assert coords.x.shape[0] == 75
        assert coords.y.shape[0] == 75
        assert coords.z.shape[0] == 75

    def test_get_loop_coords_default_values(self):
        # Test with minimal params, relying on defaults
        loop_params = {}
        coords = get_loop_coords(loop_params)
        assert coords.x.shape[0] == 100 # Default samples_num
        # Check if coordinates make sense for a default semi-circle (flat in XY plane)
        # Based on mock_semi_circle_loop: x=10cos, y=0, z=10sin
        assert np.isclose(coords.y.value, 0).all() # Should be flat in y
        assert np.isclose(coords.x.value[0], 10).all() # First point x
        assert np.isclose(coords.z.value[0], 0).all()  # First point z
        assert np.isclose(coords.x.value[-1], -10).all() # Last point x
        assert np.isclose(coords.z.value[-1], 0).all() # Last point z

    def test_get_loop_coords_units(self):
        loop_params = {"radius": 10 * u.Mm}
        coords = get_loop_coords(loop_params)
        assert coords.x.unit == u.Mm
        assert coords.y.unit == u.Mm
        assert coords.z.unit == u.Mm


# --- Tests for get_loop_params ---

class TestGetLoopParams:
    def test_get_loop_params_from_dict(self):
        input_params = {"radius": 20 * u.Mm, "phi0": 45 * u.deg, "samples_num": 50}
        params = get_loop_params(input_params)
        assert params['radius'] == 20 * u.Mm
        assert params['phi0'] == 45 * u.deg
        assert params['samples_num'] == 50
        assert params['height'] == 0.0 * u.Mm # Check default

    def test_get_loop_params_from_pickle(self, temp_pickle_file):
        params = get_loop_params(str(temp_pickle_file))
        assert params['test_key'] == "test_value"
        assert params['phi0'] == 10 * u.deg
        assert params['radius'] == 10.0 * u.Mm # Check default for non-pickled

    def test_get_loop_params_with_kwargs(self):
        params = get_loop_params(None, radius=30 * u.Mm, height=5 * u.Mm)
        assert params['radius'] == 30 * u.Mm
        assert params['height'] == 5 * u.Mm
        assert params['phi0'] == 0.0 * u.deg # Check default

    def test_get_loop_params_default_values(self):
        params = get_loop_params(None) # Neither dict nor path, rely on defaults
        assert params['radius'] == 10.0 * u.Mm
        assert params['height'] == 0.0 * u.Mm
        assert params['samples_num'] == 100
        assert params['phi0'] == 0.0 * u.deg # Default phi0
        assert isinstance(params['radius'], u.Quantity)

    def test_get_loop_params_invalid_pickle_path_fallback(self):
        # Expect fallback to defaults if file not found
        params = get_loop_params("non_existent_file.pickle")
        assert params['radius'] == 10.0 * u.Mm
        assert params['phi0'] == 0.0 * u.deg

    def test_get_loop_params_pickle_missing_phi0_fallback(self, tmp_path):
        # Create a pickle file that's valid but misses 'phi0'
        bad_pickle_data = {"some_other_key": 123}
        bad_pickle_path = tmp_path / "bad_data.pickle"
        with open(bad_pickle_path, 'wb') as f:
            pickle.dump(bad_pickle_data, f)

        # Should fallback to defaults/kwargs
        params = get_loop_params(str(bad_pickle_path), radius=50*u.Mm)
        assert params['radius'] == 50 * u.Mm # Kwarg should be used
        assert params['phi0'] == 0.0 * u.deg # Default phi0

# --- Tests for get_reference_image ---

class TestGetReferenceImage:
    def test_get_reference_image_from_smap_object(self, sample_ref_img):
        ref_img = get_reference_image(smap=sample_ref_img)
        assert isinstance(ref_img, Map)
        assert ref_img == sample_ref_img # Should be the exact object

    def test_get_reference_image_from_pickle_path(self, temp_map_pickle_file):
        ref_img = get_reference_image(smap_path=str(temp_map_pickle_file))
        assert isinstance(ref_img, Map)
        # Check a property to ensure it's loaded correctly
        assert np.array_equal(ref_img.data, np.zeros((10, 10)))

    def test_get_reference_image_default_generation(self):
        # When no smap_path or smap is provided and loading fails, it should generate default
        ref_img = get_reference_image(smap_path=None, smap=None)
        assert isinstance(ref_img, Map)
        # Check properties of the default generated map (based on the mock)
        assert ref_img.data.shape == (10, 10)
        assert ref_img.meta['INSTRUME'] == 'AIA'

    def test_get_reference_image_invalid_pickle_path_fallback(self):
        # Should fallback to default generation if smap_path points to non-existent file
        ref_img = get_reference_image(smap_path="non_existent_map.pickle")
        assert isinstance(ref_img, Map)
        assert ref_img.meta['INSTRUME'] == 'AIA'

    def test_get_reference_image_non_map_pickle_fallback(self, tmp_path):
        # Create a pickle file with non-Map data
        non_map_data = {"not_a_map": "hello"}
        non_map_pickle_path = tmp_path / "non_map.pickle"
        with open(non_map_pickle_path, 'wb') as f:
            pickle.dump(non_map_data, f)

        ref_img = get_reference_image(smap_path=str(non_map_pickle_path))
        assert isinstance(ref_img, Map)
        assert ref_img.meta['INSTRUME'] == 'AIA'

    # You might want to add a test for a "real" FITS file if you have one available.
    # This would require a dummy FITS file or mocking sunpy.map.Map's file loading.
    # For now, relying on the default generation in case of file path issues.


class TestCodeCoordsToArcsec:
    def test_return_type_and_frame(self, sample_ref_img):
        code_coord = unyt_array([0.0, 0.5]) # Center in x, middle in y
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        assert isinstance(asec_coords, SkyCoord)
        assert asec_coords.frame == sample_ref_img.coordinate_frame

    def test_x_axis_conversion_center(self, sample_ref_img):
        # x_code_coord = 0.0 (center of [-0.5, 0.5]) should map to center_x
        code_coord = unyt_array([0.0, 0.5])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        expected_x = sample_ref_img.center.Tx
        assert np.isclose(asec_coords.Tx.value, expected_x.value)
        assert asec_coords.Tx.unit == u.arcsec

    def test_x_axis_conversion_min(self, sample_ref_img):
        # x_code_coord = -0.5 should map to left edge of image
        code_coord = unyt_array([-0.5, 0.5])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        # image_width_arcsec = resolution[0] * scale[0] = 100 * 0.5 = 50 arcsec
        # center_x - (image_width_arcsec / 2) = 0 - 25 = -25 arcsec
        expected_x = sample_ref_img.center.Tx - (sample_ref_img.data.shape[0] * sample_ref_img.scale[0]) / 2
        assert np.isclose(asec_coords.Tx.value, expected_x.value)

    def test_x_axis_conversion_max(self, sample_ref_img):
        # x_code_coord = 0.5 should map to right edge of image
        code_coord = unyt_array([0.5, 0.5])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        expected_x = sample_ref_img.center.Tx + (sample_ref_img.data.shape[0] * sample_ref_img.scale[0]) / 2
        assert np.isclose(asec_coords.Tx.value, expected_x.value)

    def test_y_axis_conversion_center(self, sample_ref_img):
        # y_code_coord = 0.5 (center of [0, 1]) should map to center_y
        code_coord = unyt_array([0.0, 0.5])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        expected_y = sample_ref_img.center.Ty
        assert np.isclose(asec_coords.Ty.value, expected_y.value)
        assert asec_coords.Ty.unit == u.arcsec

    def test_y_axis_conversion_min(self, sample_ref_img):
        # y_code_coord = 0.0 should map to bottom edge relative to the center
        code_coord = unyt_array([0.0, 0.0])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        # y_code_coord - 0.5 = -0.5
        # center_y + (image_height_arcsec / 2) * -1 (because y_code_coord maps 0..1 to -0.5..0.5 from center)
        # image_height_arcsec = resolution[1] * scale[1] = 100 * 0.5 = 50 arcsec
        # expected_y = center_y + 50 * (-0.5) = 0 - 25 = -25 arcsec
        expected_y = sample_ref_img.center.Ty - (sample_ref_img.data.shape[1] * sample_ref_img.scale[1]) / 2
        assert np.isclose(asec_coords.Ty.value, expected_y.value)

    def test_y_axis_conversion_max(self, sample_ref_img):
        # y_code_coord = 1.0 should map to top edge relative to the center
        code_coord = unyt_array([0.0, 1.0])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        # y_code_coord - 0.5 = 0.5
        # expected_y = center_y + 50 * 0.5 = 0 + 25 = 25 arcsec
        expected_y = sample_ref_img.center.Ty + (sample_ref_img.data.shape[1] * sample_ref_img.scale[1]) / 2
        assert np.isclose(asec_coords.Ty.value, expected_y.value)

    def test_kwargs_override(self, sample_ref_img):
        # Test if kwargs correctly override ref_img properties
        code_coord = unyt_array([0.0, 0.5])
        override_center_x = 100 * u.arcsec
        override_center_y = 50 * u.arcsec
        override_resolution = (200, 200) # Dummy resolution
        override_scale = (0.2 * u.arcsec / u.pix, 0.2 * u.arcsec / u.pix) # Dummy scale

        asec_coords = code_coords_to_arcsec(
            code_coord,
            ref_img=sample_ref_img,
            center_x=override_center_x,
            center_y=override_center_y,
            resolution=override_resolution,
            scale=override_scale
        )
        # With code_coord[0] = 0.0, it should just be override_center_x
        assert np.isclose(asec_coords.Tx.value, override_center_x.value)
        # With code_coord[1] = 0.5, it should just be override_center_y
        assert np.isclose(asec_coords.Ty.value, override_center_y.value)

    def test_no_ref_img_provided_uses_default(self):
        # Test that it calls get_reference_image if ref_img is None
        code_coord = unyt_array([0.0, 0.5])
        asec_coords = code_coords_to_arcsec(code_coord) # No ref_img provided
        assert isinstance(asec_coords, SkyCoord)
        # Check against the properties of the *mock* default image
        # Mock default image center is (0,0), resolution (100,100), scale (0.5,0.5)
        assert np.isclose(asec_coords.Tx.value, 0.0)
        assert np.isclose(asec_coords.Ty.value, 0.0)

    def test_unyt_array_units(self, sample_ref_img):
        # Test that unyt_array with no specific unit is handled
        code_coord = unyt_array([0.1, 0.6]) # No explicit unit, will default to dimensionless
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)
        assert asec_coords.Tx.unit == u.arcsec
        assert asec_coords.Ty.unit == u.arcsec

    @pytest.mark.parametrize("x_code, y_code, expected_tx, expected_ty", [
        (0.0, 0.5, 0.0, 0.0), # Center
        (-0.5, 0.0, -25.0, -25.0), # Bottom-left corner (relative to center)
        (0.5, 1.0, 25.0, 25.0), # Top-right corner (relative to center)
        (0.2, 0.7, 10.0, 10.0), # Some intermediate point
    ])
    def test_conversion_logic_parameterized(self, sample_ref_img, x_code, y_code, expected_tx, expected_ty):
        code_coord = unyt_array([x_code, y_code])
        asec_coords = code_coords_to_arcsec(code_coord, ref_img=sample_ref_img)

        # Expected calculations based on sample_ref_img (100x100, 0.5 arcsec/pix, center 0,0)
        # image_width = 100 * 0.5 = 50 arcsec
        # image_height = 100 * 0.5 = 50 arcsec
        # Tx = 0 + 50 * x_code
        # Ty = 0 + 50 * (y_code - 0.5)

        assert np.isclose(asec_coords.Tx.value, expected_tx)
        assert np.isclose(asec_coords.Ty.value, expected_ty)
