import re

def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_cfg(path,is_num=False):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        if is_num:
            value = value.strip()
            if is_number(value):
                if value.isdigit():
                    options[key.strip()] = int(value)
                else:
                    options[key.strip()] = float(value)
            else:
                options[key.strip()] = value.strip()
        else:
            options[key.strip()] = value.strip()
    return options

def parse_data_name(path):
    """Parses the data names"""
    name = ''
    with open(path, 'r') as fp:
        lines = fp.read()
    for line in lines:
        line = line.strip()
        name = name+line
    return name

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict,self).__init__(*args, **kwargs)
        self.__dict__ = self

def parse_dict2params(path):
    return AttrDict(parse_data_cfg(path,is_num=True))

def parse_params2dict(opt,config):
    opt = vars(opt)
    for key in opt:
        if key not in config:
            config[key] = opt[key]

def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False