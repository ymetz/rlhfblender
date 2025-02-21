"""
Read parsing of args-config files from SB3 to enable automatic model loading possible
"""

import yaml

yaml.add_multi_constructor("!!", lambda loader, suffix, node: None)
yaml.add_multi_constructor("tag:yaml.org,2002:python/name", lambda loader, suffix, node: None, Loader=yaml.SafeLoader)


def read_sb3_configs(path: str) -> dict:

    with open(path) as theyaml:
        try:
            _ = theyaml.readline()  # sb3 has a header line in the yaml file
            yaml_content = yaml.safe_load(theyaml)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    config = {}
    for elem in yaml_content[0]:
        config[elem[0]] = elem[1]

    return config
