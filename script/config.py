import os
import re
import yaml
from easydict import EasyDict as edict

def scientific_notation_constructor(loader, node):
    value = loader.construct_scalar(node)
    match = re.match(r'^([+-]?(?:\d+\.?\d*|\.\d+))(?:[eE]([+-]?\d+))?$', value)
    if match:
        return float(value)
    return value

yaml.add_implicit_resolver('!float', re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))(?:[eE]([+-]?\d+))?$'), first='-.0123456789')
yaml.add_constructor('!float', scientific_notation_constructor)

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def get_config(config_file=None):
    return YamlParser(config_file=config_file)