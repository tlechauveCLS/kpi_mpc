import glob
from configparser import ConfigParser
from datetime import datetime

import numpy as np
import os
from pandas import DataFrame

from mpc.kpi1b.parsers import SATELLITE_LIST, WV_LIST

LOG_FORMAT = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'


def get_parameter(name, config: ConfigParser, args, required=False, format=None):
    value = None

    if name in args:
        value = getattr(args, name)

    if value is None and 'DEFAULT' in config and name in config['DEFAULT']:
        value = config.get('DEFAULT', name)

    if value is None and required:
        raise AttributeError(f'Missing {name} parameter.')

    if value is not None and format:
        value = format(value)

    return value


def get_dataframe(inputs_dir, dt: datetime = None):
    data = []
    cpt_corrupt = 0

    for sat in SATELLITE_LIST:
        for wv in WV_LIST:
            if dt:
                day = dt.strftime('%Y%m%d')
            else:
                day = '20*'
            inputs = sorted(glob.glob(os.path.join(inputs_dir, f'kpi_output_{sat}_{wv}_{day}.txt')))

            for input in inputs:

                with open(input) as f:
                    content = f.readlines()[0].replace('\n', '').split(' ')

                if len(content) == 7:
                    kpix, stax, _, stox, _, envx, nbx = content
                    meanbias = std = np.nan
                elif len(content) == 9:
                    kpix, stax, _, stox, _, envx, nbx, meanbias, std = content
                else:
                    cpt_corrupt += 1
                    continue

                data.append({'sat': sat,
                             'wv': wv,
                             'kpi': float(kpix),
                             'sta': datetime.strptime(stax, '%Y-%m-%d'),
                             'sto': datetime.strptime(stox, '%Y-%m-%d'),
                             'env': float(envx),
                             'nb': int(nbx),
                             'bias': float(meanbias),
                             'std': float(std)})

    return DataFrame(data)
