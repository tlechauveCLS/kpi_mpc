# -*- coding: utf-8 -*-
"""
IFREMER
Oct 2021
KPI-1b SLA Vol3 document MPC contract 2021 - 2016
"""
import datetime
import logging
import sys
import textwrap
import time
import argparse
from configparser import ConfigParser

import resource
from dateutil.parser import parse as parse_date
import numpy as np
import os
from mpc.kpi1b.reader import read_fat_calib_nc, SATELLITE_LIST

POLARIZATION = 'VV'
MODE = 'WV'
# ENVELOP = 2 #sigma
ENVELOP = 95  # %
PRIOR_PERIOD = 3  # months
LAT_MAX = 55
MIN_DIST_2_COAST = 100  # km

WV_LIST = ['wv1', 'wv2']

log = logging.getLogger('mpc.compute_kpi_1b')

LOG_FORMAT = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'


class KPI1BData:

    def __init__(self, value: float, start: datetime, stop: datetime, envelop_value: float, count: int, count_in: int,
                 mean_bias: float, std: float):
        """
        :param value: float between 0 and 100 %
        :param start: Start of the current month
        :param stop: Stop datetime of the current month
        :param envelop_value: 2-sigma dB threshold based on 3 months prior period
        :param count: Total number of measure
        :param count_in: Total number of measure inside the envelope
        :param mean_bias:
        :param std:
        """
        self.value = value
        self.start = start
        self.stop = stop
        self.envelop_value = envelop_value
        self.count = count
        self.count_in = count_in
        self.mean_bias = mean_bias
        self.std = std

    def __repr__(self):
        return f'{self.value} {self.start} {self.stop} {self.envelop_value} {self.count} {self.count_in} {self.mean_bias} {self.std}'


def compute_kpi_1b(sat: str, wv: str, input_path_pattern: str, coastline_netcdf: str,
                   stop_analysis_period: datetime = None, df_slc_sat=None):
    """
    NRCS (denoised) observed compared to predicted GMF CMOD5n
    :param sat: S1A or ..
    :param wv: wv1 or wv2
    :param stop_analysis_period: datetime (-> period considered date-1 month : date)
    :param df_slc_sat:
    :return:
        a KPI_1B object

    """
    if df_slc_sat is None:
        df_slc_sat = read_fat_calib_nc(input_path_pattern, coastline_netcdf, satellite_list=[sat])

    if stop_analysis_period is None:
        stop_current_month = datetime.datetime.today()
    else:
        stop_current_month = stop_analysis_period

    start_current_month = stop_current_month - datetime.timedelta(days=30)
    log.debug('start_current_month : %s', start_current_month)
    log.debug('stop_current_month : %s', stop_current_month)
    # compute the 2 sigma envelopp on the last 3 months prior to current month
    start_prior_period = start_current_month - datetime.timedelta(days=30 * PRIOR_PERIOD)
    stop_prior_period = start_current_month
    df_slc = df_slc_sat[sat]
    df_slc['direct_diff_calib_cst_db'] = df_slc['sigma0_denoised_db'] - df_slc['tmp_gmf_cmod5n_nrcs_db']
    if wv == 'wv1':
        cond_inc = (df_slc['_inc_sar'] < 30)
    elif wv == 'wv2':
        cond_inc = (df_slc['_inc_sar'] > 30)
    else:
        raise Exception('wv value un expected : %s' % wv)
    ocean_acqui_filters = (abs(df_slc['_lat_sar']) < LAT_MAX) & (df_slc['_land'] == False) & (
            df_slc['distance2coast'] > MIN_DIST_2_COAST)
    mask_prior_period = ocean_acqui_filters & cond_inc & (df_slc['time'] >= start_prior_period) & (
            df_slc['time'] <= stop_prior_period) & (np.isfinite(df_slc['direct_diff_calib_cst_db']))
    subset_df = df_slc[mask_prior_period]
    nb_nan = np.isnan(subset_df['direct_diff_calib_cst_db']).sum()
    log.debug('some values: %s', subset_df['direct_diff_calib_cst_db'].values)
    log.info('nb_nan : %s', nb_nan)
    log.info('nb finite %s/%s', np.isfinite(subset_df['direct_diff_calib_cst_db']).sum(),
             len(subset_df['direct_diff_calib_cst_db']))
    # envelop_value = ENVELOP*np.nanstd(subset_df['direct_diff_calib_cst_db'])
    std = np.nanstd(subset_df['direct_diff_calib_cst_db'])
    envelop_value = np.percentile(abs(subset_df['direct_diff_calib_cst_db']), ENVELOP)
    log.debug('envelop_value : %s', envelop_value)

    # compute the number of product within the envelop for current month
    nb_measu_total = 0
    nb_measu_inside_envelop = 0
    mask_current_month = ocean_acqui_filters & cond_inc & (df_slc['time'] >= start_current_month) & (
            df_slc['time'] <= stop_current_month)
    subset_current_period = df_slc[mask_current_month]
    log.info('nb pts current month : %s', len(subset_current_period['time']))
    nb_measu_total = len(subset_current_period['time'])
    # nb_measu_outside_envelop = (abs(subset_current_period['direct_diff_calib_cst_db'])<envelop_value).sum()

    # definition proposee par Hajduch le 10dec2021 screenshot a lappuit (je ne suis pas convaincu pas l introduction du biais dans le calcul de levenveloppe car le KPI sera dautant plus elever que le biais sera fort (cest linverse qui est cherche)
    bias_minus_2sigma = abs(subset_current_period['direct_diff_calib_cst_db'].mean() - envelop_value)
    bias_plus_2sigma = abs(subset_current_period['direct_diff_calib_cst_db'].mean() + envelop_value)
    log.info('bias_plus_2sigma %s', bias_plus_2sigma)
    T = np.max([bias_minus_2sigma, bias_plus_2sigma])
    log.info('T : %s %s', T.shape, T)
    nb_measu_inside_envelop = (abs(subset_current_period['direct_diff_calib_cst_db']) < T).sum()
    mean_bias = np.mean(subset_current_period['direct_diff_calib_cst_db'])
    log.debug('nb_measu_inside_envelop : %s', nb_measu_inside_envelop)
    kpi_value = 100. * nb_measu_inside_envelop / nb_measu_total

    log.debug('kpi_value : %s', kpi_value)

    return KPI1BData(kpi_value, start_current_month, stop_current_month, envelop_value, nb_measu_total,
                     nb_measu_inside_envelop, mean_bias, std)


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''
    Compute KPI-1B and save results as textfile
    
    Use configuration file (--config) to avoid a long list of parameters. Example; kpi1b.ini
    
       [DEFAULT]
        output=/tmp/output
        coastline_netcdf=/tmp/NASA_tiff_distance_to_coast_converted_v2.nc
        input_path_pattern=/tmp/%%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc
        
    ! Careful inline arguments in command will overwrite its value in configuration file.
    '''))
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-f', '--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=%(default)s]')

    parser.add_argument('-c', '--config', default=os.getenv('MPC_KPI1B_CONFIG'),
                        help='Path to the configuration file.')

    parser.add_argument('--enddate', help='end of the 1 month period analysed')
    parser.add_argument('-o', '--output', help='Path output will be stored')
    parser.add_argument('--coastline-netcdf', help='Coastline NetCDF')
    parser.add_argument('--inputs-path-pattern', help='Path to input files as pattern, the satellite unit will be'
                                                      'passed as parameter for this pattern. Use %%s where the unit'
                                                      'should be placed in the filename. Use %%%%s in the configfile')

    parser.add_argument('satellite', choices=SATELLITE_LIST,
                        help='S-1 unit choice')
    parser.add_argument('wv', choices=WV_LIST,
                        help='WV incidence angle choice')
    return parser


def setup_log(debug=False):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    log.setLevel(logging.DEBUG if debug else logging.INFO)
    log.addHandler(handler)
    log.propagate = False


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


def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_log(args.verbose)

    t0 = time.time()
    sat = args.satellite
    wv = args.wv

    config = ConfigParser()
    if args.config:
        config.read(args.config)

    output = get_parameter('output', config, args, required=True)
    coastline_netcdf = get_parameter('coastline_netcdf', config, args, required=True)
    input_path_pattern = get_parameter('input_path_pattern', config, args, required=True)
    end_date = get_parameter('enddate', config, args, format=parse_date)

    output_file = os.path.join(output, f'kpi_output_{sat}_{wv}_{end_date if end_date else ""}.txt')

    if os.path.exists(output_file) and not args.overwrite:
        log.info(f'output {output_file} already exists')
    else:
        kpi = compute_kpi_1b(sat, wv, input_path_pattern, coastline_netcdf, stop_analysis_period=end_date)
        log.info('#' * 10)
        log.info(f'kpi_v {sat} {wv} : {kpi.value} (envelop {ENVELOP}-sigma value: {kpi.envelop_value} dB)')
        log.info('#' * 10)
        log.info(f'start_cur_month : {kpi.start}, stop_cur_month : {kpi.stop}')

        os.makedirs(os.path.dirname(output_file), 0o0775, exist_ok=True)

        with open(output_file, 'w') as fid:
            fid.write(str(kpi))

        log.info(f'output: {output_file}')
        log.info(f'done in {(time.time() - t0) / 60.:1.3f} min')
        log.info(f'peak memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.} Mbytes')

    log.info("over")


if __name__ == '__main__':
    main()
