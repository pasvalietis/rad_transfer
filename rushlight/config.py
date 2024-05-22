import os
import yaml
import json
import glob
import ast
from configparser import ConfigParser
from dataclasses import dataclass

'''
Configuration file containing required paths, that can be later exported into any other parts of code 
'''

#%%
env_vars = os.environ
# Locate configuration file
config_data = None
config_parser = ConfigParser()
pkg_dir = os.path.dirname(os.path.abspath(__file__))

for loc in pkg_dir, os.path.expanduser("~"), os.environ.get('CONDA_PREFIX'), os.getcwd():
    try:
        with open(os.path.join(loc, "conf.ini")) as source:
            config_parser.read_file(source)

        config_data = config_parser

    except FileNotFoundError:
        # Check if configuration data is stored in json or yaml files in listed directories
        # Try to load information about nested directory structures
        conf_files = [('*conf*.json', json.load), ('*conf*.yaml', yaml.safe_load)]
        try:
            for pattern, load_function in conf_files:
                conf_path = glob.glob(os.path.join(loc, pattern))
                if conf_path:
                    with open(conf_path[0]) as source:
                        config_data = load_function(source)
        except IOError:
            pass

if config_data is None:
    raise FileNotFoundError('Configuration file is not found.')

#%%
@dataclass
class Config:
    PACKAGE_DIR: str
    JSOC_EMAIL: str
    INSTRUMENTS: dict
    OBSERVATIONS: dict
    SIMULATIONS: dict

    def __init__(self):
        self.PACKAGE_DIR = pkg_dir

        # Acquiring data
        self.JSOC_EMAIL = config_data['DEFAULT']['JSOC_EMAIL']

        self.INSTRUMENTS = config_data['INSTRUMENTS']
        self.OBSERVATIONS = config_data['OBSERVATIONS']
        self.SIMULATIONS = config_data['SIMULATIONS']

        self.DIR_STRUCT = ast.literal_eval(config_data['OBSERVATIONS']['DIR_STRUCT'])

        # Temperature response functions
        # INSTR_TEMP_RESPONSE

        # Databases etc.
        # e.g. Chianti database

# Check if a config file exists and create an instance of Config class to store all paths
#%%
config = Config()
