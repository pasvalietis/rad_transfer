import pytest
import rushlight.utils.proj_imag_classified as pic


@pytest.mark.xfail(reason="Testing class for proj_imag_classified not finalized")
class TestSyntheticFilterImage:
    # This method is called before EACH test_ method in this class

    @pytest.mark.parametrize("dataset, smap_path, smap, kwargs", [
        (0, 5, 5, 1),
        (10, 20, 30, 1),
        (-5, 5, 0, 1),
        (0, 0, 0, 1)
    ])
    def setup_method(self, method, dataset, smap_path, smap, kwargs):
        print(f"\nSetting up for {method.__name__}")
        self.sfiObj = pic.SyntheticFilterImage(dataset, smap_path, smap, **kwargs) # Each test gets a fresh instance

    # This method is called after EACH test_ method in this class
    def teardown_method(self, method):
        print(f"Tearing down after {method.__name__}")
        del self.sfiObj # Clean up resources if necessary

    def test_set_loop_params(self):
        self.calculator.add(25)
        assert self.calculator.get_current_value() == 125

    @classmethod
    def setup_class(cls):
        print("\nSetting up TestSyntheticFilterImage class (once)")
        # This is good for setting up something shared across ALL tests in this class,
        # but be careful about modifying shared state if tests aren't isolated.

    @classmethod
    def teardown_class(cls):
        print("Tearing down TestSyntheticFilterImage class (once)")