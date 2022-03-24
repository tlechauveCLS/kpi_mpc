from configparser import ConfigParser

SATELLITE_LIST = ['S1A', 'S1B']
WV_LIST = ['wv1', 'wv2']
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
