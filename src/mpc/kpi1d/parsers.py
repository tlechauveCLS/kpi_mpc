import calendar
from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime

import os
from dateutil.parser import parse as parse_date
from logbook import Logger

DEFAULT_CONFIG = os.getenv('MPC_KPI1D_CONFIG')

DEFAULT_SATELLITE = 'S1A'
DEFAULT_WV = 'wv1'
DEFAULT_OUTPUT = '/tmp/kpi1d/output'
DEFAULT_L2F_PATH = '/tmp/kpi1d/l2f'
DEFAULT_OVERWRITE = False

SATELLITE_LIST = ['S1A', 'S1B']
WV_LIST = ['wv1', 'wv2']

log = Logger('mpc.kpi1d.parsers')


def compute_config_parser(args):
    config = ConfigParser()
    config.read(args.ini)

    satellite = config.get('DEFAULT', 'satellite', fallback=DEFAULT_SATELLITE)
    wv = config.get('DEFAULT', 'wv', fallback=DEFAULT_WV)

    if args.enddate:
        enddate = args.enddate
    else:
        dt = datetime.utcnow()
        _, last_day = calendar.monthrange(dt.year, dt.month)
        enddate = config.get('DEFAULT', 'enddate',
                             fallback=dt.replace(day=last_day, hour=23, minute=59, second=59).strftime('%Y%m%dT%H%M%S'))
    enddate = parse_date(enddate)
    output = config.get('DEFAULT', 'output', fallback=DEFAULT_OUTPUT)
    l2f_path = config.get('DEFAULT', 'l2f_path', fallback=DEFAULT_L2F_PATH)
    overwrite = config.get('DEFAULT', 'overwrite', fallback=DEFAULT_OVERWRITE)

    return {
        'satellite': satellite,
        'wv': wv,
        'enddate': enddate,
        'output': output,
        'l2f_path': l2f_path,
        'overwrite': overwrite
    }


def cli_parser(args):
    return vars(args)


def compute_parser():
    parser = ArgumentParser()

    parser.add_argument('-v', '--verbose', dest='debug', help='Enable debug LOG', action='store_true', default=False)
    parser.add_argument('--enddate', help='end of the 1 month period analysed')

    subparsers = parser.add_subparsers()

    confp = subparsers.add_parser('config')
    confp.add_argument('-i', '--ini', help='Path to the .ini configuration file', default=DEFAULT_CONFIG)
    confp.add_argument('-e', '--enddate', help='end of the 1 month period analysed')
    confp.set_defaults(func=compute_config_parser)

    clip = subparsers.add_parser('query')
    clip.add_argument('-e', '--enddate', help='end of the 1 month period analysed')
    clip.add_argument('-o', '--output', help='Path output will be stored')
    clip.add_argument('-f', '--overwrite', help='Overwrite existent output')
    parser.add_argument('--l2f-path', help='Path to directory for L2F files')
    clip.add_argument('satellite', choices=SATELLITE_LIST, help='S-1 unit choice')
    clip.add_argument('wv', choices=WV_LIST, help='WV incidence angle choice')

    clip.set_defaults(func=cli_parser)

    return parser
