import pytest
import rushlight.utils.proj_imag_classified as sfi

class TestSyntheticFilterImage:
    # This method is called before EACH test_ method in this class

    def setup_method(self, method):
        print(f"\nSetting up for {method.__name__}")
        self.sfiObj = sfi.SyntheticFilterImage() # Each test gets a fresh instance

    # This method is called after EACH test_ method in this class
    def teardown_method(self, method):
        print(f"Tearing down after {method.__name__}")
        del self.sfiObj # Clean up resources if necessary

    @classmethod
    def setup_class(cls):
        print("\nSetting up TestSyntheticFilterImage class (once)")
        # This is good for setting up something shared across ALL tests in this class,
        # but be careful about modifying shared state if tests aren't isolated.

    @classmethod
    def teardown_class(cls):
        print("Tearing down TestSyntheticFilterImage class (once)")