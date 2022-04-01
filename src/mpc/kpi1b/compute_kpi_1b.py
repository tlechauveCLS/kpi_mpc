# -*- coding: utf-8 -*-
"""
IFREMER
Oct 2021
KPI-1b SLA Vol3 document MPC contract 2021 - 2016
"""
import datetime
import logging
import sys
import time

import numpy as np
import os
import resource

from mpc.common import LOG_FORMAT
from mpc.kpi1b.parsers import compute_parser
from mpc.kpi1b.reader import read_fat_calib_nc

POLARIZATION = 'VV'
MODE = 'WV'
# ENVELOP = 2 #sigma
ENVELOP = 95  # %
PRIOR_PERIOD = 3  # months
LAT_MAX = 55
MIN_DIST_2_COAST = 100  # km

log = logging.getLogger('mpc.compute_kpi_1b')


class KPI1BItem:

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


def compute_kpi_1b(satellite: str, wv: str, inputs_path_pattern: str, coastline_netcdf: str,
                   enddate: datetime = None, df_slc_sat=None, output=None, overwrite=False):
    """
    NRCS (denoised) observed compared to predicted GMF CMOD5n
    :param sat: S1A or ..
    :param wv: wv1 or wv2
    :param enddate: datetime (-> period considered date-1 month : date)
    :param df_slc_sat:
    :return:
        a KPI_1B object

    """
    if df_slc_sat is None:
        df_slc_sat = read_fat_calib_nc(inputs_path_pattern, coastline_netcdf, satellite_list=[satellite])

    if enddate is None:
        stop_current_month = datetime.datetime.today()
    else:
        stop_current_month = enddate

    start_current_month = stop_current_month - datetime.timedelta(days=30)
    log.debug('start_current_month : %s', start_current_month)
    log.debug('stop_current_month : %s', stop_current_month)
    # compute the 2 sigma envelopp on the last 3 months prior to current month
    start_prior_period = start_current_month - datetime.timedelta(days=30 * PRIOR_PERIOD)
    stop_prior_period = start_current_month
    df_slc = df_slc_sat[satellite]
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

    item = KPI1BItem(kpi_value, start_current_month, stop_current_month, envelop_value, nb_measu_total,
                     nb_measu_inside_envelop, mean_bias, std)

    if output:
        output_file = os.path.join(output, f'kpi_output_{satellite}_{wv}_{enddate if enddate else ""}.txt')
        os.makedirs(os.path.dirname(output_file), 0o0775, exist_ok=True)

        if os.path.exists(output_file) and not overwrite:
            log.info(f'output {output_file} already exists, skip writing')
        else:
            with open(output_file, 'w') as fid:
                fid.write(str(item))

            log.info(f'output: {output_file}')

    return item


def setup_log(debug=False):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    log.setLevel(logging.DEBUG if debug else logging.INFO)
    log.addHandler(handler)
    log.propagate = False


def main():
    parser = compute_parser()
    args = parser.parse_args()
    setup_log(args.debug)

    params = args.func(args)

    t0 = time.time()

    kpi = compute_kpi_1b(**params)
    log.info('#' * 10)
    log.info(
        f'kpi_v {params["satellite"]} {params["wv"]} : {kpi.value} (envelop {ENVELOP}-sigma value: {kpi.envelop_value} dB)')
    log.info('#' * 10)
    log.info(f'start_cur_month : {kpi.start}, stop_cur_month : {kpi.stop}')

    log.info(f'done in {(time.time() - t0) / 60.:1.3f} min')
    log.info(f'peak memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.} Mbytes')

    log.info("over")


if __name__ == '__main__':
    main()
