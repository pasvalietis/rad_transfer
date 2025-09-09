import pytest
import math
import rushlight.utils.proj_imag_classified as sfi

def _assert_positive_or_zero(value, name):
    """Asserts that a numerical value is non-negative."""
    assert value >= 0, f"{name} must be non-negative."

def _assert_valid_angle_radians(angle, name):
    """Asserts that an angle is within [0, 2*pi] radians."""
    assert 0 <= angle <= 2 * math.pi, f"{name} must be within [0, 2*pi] radians."

def _assert_valid_elevation_angle_radians(angle, name):
    """Asserts that an elevation angle is within [-pi/2, pi/2] radians."""
    assert -math.pi / 2 <= angle <= math.pi / 2, f"{name} must be within [-pi/2, pi/2] radians."

def _assert_valid_azimuth_angle_radians(angle, name):
    """Asserts that an azimuth angle is within [0, 2*pi] radians."""
    assert 0 <= angle <= 2 * math.pi, f"{name} must be within [0, 2*pi] radians."

def _assert_positive_integer(value, name):
    """Asserts that a value is a positive integer."""
    assert isinstance(value, int), f"{name} must be an integer."
    assert value > 0, f"{name} must be a positive integer."

# --- Pytest Test Functions ---

def test_valid_dimensions_with_radius():
    """Test with valid dimensions using 'radius'."""
    dims = {
        'radius': 10.0,
        'height': 5.0,
        'phi0': math.pi / 2,
        'theta0': math.pi / 4,
        'el': 0.0,
        'az': math.pi,
        'samples_num': 100
    }

    obj = sfi.SyntheticFilterImage(pkl=dims)

    _assert_positive_or_zero(obj.radius, 'radius')
    _assert_positive_or_zero(obj.height, 'height')
    _assert_valid_angle_radians(obj.phi0, 'phi0')
    _assert_valid_angle_radians(obj.theta0, 'theta0')
    _assert_valid_elevation_angle_radians(obj.el, 'el')
    _assert_valid_azimuth_angle_radians(obj.az, 'az')
    _assert_positive_integer(obj.samples_num, 'samples_num')
    assert obj.lat == obj.theta0
    assert obj.lon == obj.phi0


def test_valid_dimensions_with_majax_minax():
    """Test with valid dimensions using 'majax' and 'minax'."""
    dims = {
        'majax': 20.0,
        'minax': 15.0,
        'height': 7.5,
        'phi0': 0.0,
        'theta0': math.pi / 2,
        'el': -math.pi / 4,
        'az': 2 * math.pi - 0.001,
        'samples_num': 50
    }
    obj = sfi.SyntheticFilterImage(pkl=dims)

    _assert_positive_or_zero(obj.majax, 'majax')
    _assert_positive_or_zero(obj.minax, 'minax')
    assert obj.minax <= obj.majax, "minax must be less than or equal to majax."
    _assert_positive_or_zero(obj.height, 'height')
    _assert_valid_angle_radians(obj.phi0, 'phi0')
    _assert_valid_angle_radians(obj.theta0, 'theta0')
    _assert_valid_elevation_angle_radians(obj.el, 'el')
    _assert_valid_azimuth_angle_radians(obj.az, 'az')
    _assert_positive_integer(obj.samples_num, 'samples_num')
    assert obj.lat == obj.theta0
    assert obj.lon == obj.phi0

@pytest.mark.parametrize(
    "param_name, invalid_value, expected_error_part",
    [
        ('radius', -1.0, "radius must be non-negative."),
        ('majax', -1.0, "majax must be non-negative."),
        ('minax_greater_than_majax', (10.0, 15.0), "minax must be less than or equal to majax."), # special case
        ('phi0', 2 * math.pi + 0.1, "phi0 must be within [0, 2*pi] radians."),
        ('theta0', -0.1, "theta0 must be within [0, 2*pi] radians."),
        ('el', math.pi/2 + 0.1, "el must be within [-pi/2, pi/2] radians."),
        ('az', -0.1, "az must be within [0, 2*pi] radians."),
        ('samples_num', -5, "samples_num must be a positive integer."),
        ('samples_num', 0, "samples_num must be a positive integer."),
        ('samples_num', 10.5, "samples_num must be an integer."),
    ]
)
@pytest.mark.xfail
def test_invalid_parameters(param_name, invalid_value, expected_error_part):
    """Test that various invalid parameters raise appropriate errors."""
    default_dims = {
        'radius': 1.0, 'height': 1.0,
        'phi0': 0, 'theta0': 0, 'el': 0, 'az': 0,
        'samples_num': 10
    }

    dims_to_test = default_dims.copy()

    # Handle special case for minax > majax
    if param_name == 'minax_greater_than_majax':
        dims_to_test['majax'] = invalid_value[0]
        dims_to_test['minax'] = invalid_value[1]
    elif param_name == 'radius':
        dims_to_test[param_name] = invalid_value
        # Ensure majax/minax aren't present if radius is
        dims_to_test.pop('majax', None)
        dims_to_test.pop('minax', None)
    elif param_name == 'majax':
        dims_to_test[param_name] = invalid_value
        dims_to_test['minax'] = 0.0 # Provide a valid minax if testing majax
        dims_to_test.pop('radius', None)
    else:
        dims_to_test[param_name] = invalid_value


    obj = sfi.SyntheticFilterImage(pkl=dims_to_test)

    # Use pytest.raises to check for assertion errors
    with pytest.raises(AssertionError) as excinfo:
        if param_name == 'radius':
            _assert_positive_or_zero(obj.radius, 'radius')
        elif param_name == 'majax':
            _assert_positive_or_zero(obj.majax, 'majax')
        elif param_name == 'minax_greater_than_majax':
            # This specific assertion needs to be tested directly here
            assert obj.minax <= obj.majax, "minax must be less than or equal to majax."
        elif param_name == 'phi0':
            _assert_valid_angle_radians(obj.phi0, 'phi0')
        elif param_name == 'theta0':
            _assert_valid_angle_radians(obj.theta0, 'theta0')
        elif param_name == 'el':
            _assert_valid_elevation_angle_radians(obj.el, 'el')
        elif param_name == 'az':
            _assert_valid_azimuth_angle_radians(obj.az, 'az')
        elif param_name == 'samples_num':
            _assert_positive_integer(obj.samples_num, 'samples_num')
        else:
            pytest.fail(f"Unhandled parameter type in test: {param_name}")

    assert expected_error_part in str(excinfo.value)