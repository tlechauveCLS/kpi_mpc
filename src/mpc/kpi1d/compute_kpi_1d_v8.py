# -*- coding: utf-8 -*-
"""
Definition of KPI: decided by NORSE+CLS+IFREME
implementaiton : IFREMER
Dec 2021: after SR#3 + ORR to adjust the KPI
KPI-1d SLA Vol3 document MPC contract 2021 - 2016
"""
import datetime
import logging
import sys
import time

import numpy as np
import os
import resource
import xarray

from mpc.common import LOG_FORMAT
from mpc.kpi1d.parsers import compute_parser
from mpc.kpi1d.read_and_concat_L2F import read_L2F_with_xarray

POLARIZATION = 'VV'
MODE = 'WV'
# ENVELOP = 2 #sigma
ENVELOP = 95  # % percentile
PRIOR_PERIOD = 3  # months
LAT_MAX = 55
MIN_DIST_2_COAST = 100  # km
MS = {'wv1': -6.5, 'wv2': -15.}  # NRCS central values dB
DS = {'wv1': 3, 'wv2': 5}  # delta

log = logging.getLogger('mpc.compute_kpi_1d')


class KPI1DItem:

    def __init__(self, value: float, start: datetime, stop: datetime, envelop_value: float, count: int,
                 mean_bias: float, std: float):
        """
        :param value: float between 0 and 100 %
        :param start: Start of the current month
        :param stop: Stop datetime of the current month
        :param envelop_value: 2-sigma dB threshold based on 3 months prior period
        :param count: Total number of measure
        :param mean_bias:
        :param std:
        """
        self.value = value
        self.start = start
        self.stop = stop
        self.envelop_value = envelop_value
        self.count = count
        self.mean_bias = mean_bias
        self.std = std

    def __repr__(self):
        return f'{self.value} {self.start} {self.stop} {self.envelop_value} {self.count} {self.std}'


def load_Level2_series(satellite, start, stop, l2f_path):
    """

    :param satellite: str S1A or ...
    :return:
    """

    log.info('load L2F data')
    vv = ['oswQualityFlagPartition1', 'fdatedt', 'oswLon', 'oswLat', 'oswHeading',
          's1_effective_hs_2Dcutoff', 'ecmwf0125_uwind', 'ecmwf0125_vwind',
          'oswIncidenceAngle', 'oswLandFlag', 'dist2coastKM', 'pol', 'class_1',
          'ww3_effective_2Dcutoff_hs', 'oswNv', 'oswNrcs', 'oswAzCutoff', 'oswEcmwfWindSpeed',
          'oswQualityFlagPartition1', 'oswQualityFlagPartition2', 'oswQualityFlagPartition3',
          'oswQualityFlagPartition4', 'oswQualityFlagPartition5', 's1_hs_emp_tot_v3p2',
          'oswXA_hs_ww3spec_firstSARpartition', 'oswXA_hs_ww3spec_secondSARpartition',
          'oswXA_hs_ww3spec_thirdSARpartition',
          'oswXA_hs_ww3spec_fourthSARpartition', 'oswXA_hs_ww3spec_fifthSARpartition',
          "oswXA_wl_ww3spec_firstSARpartition", 'oswXA_wl_ww3spec_secondSARpartition',
          'oswXA_wl_ww3spec_thirdSARpartition',
          'oswXA_wl_ww3spec_fourthSARpartition', 'oswXA_wl_ww3spec_fifthSARpartition'
          ]
    ds_dict_sat = read_L2F_with_xarray(start, stop, l2f_path, satellites=[satellite], variables=vv,
                                       add_ecmwf_wind=True)
    dswv = ds_dict_sat[satellite]
    log.info('dswv type %s', type(dswv))
    if dswv != {}:
        # drop Nan
        dswv = dswv.where(
            np.isfinite(dswv['s1_effective_hs_2Dcutoff']) & np.isfinite(dswv['ww3_effective_2Dcutoff_hs']), drop=True)
    return dswv


def compute_kpi_1d(satellite, wv, l2f_path, dev=False, enddate=None, period_analysed_width=30, df=None,
                   output=None, overwrite=False):
    """
    osw VV WV S-1 effective Hs (2D-cutoff) compared to WWIII Hs computed on same grid/same mask
    note that low freq mask is applied both on S-1 spectrum and WWIII spectrum
    :param l2f_path:
    :param sat: str S1A or ..
    :param wv: str wv1 or wv2
    :param enddate: datetime (-> period considered T-1 month : T)
    :param period_analysed_width : int 30 days by default
    :return:
        kpi_value (float): between 0 and 100 %
        start_current_month (datetime):
        stop_current_month (datetime):
        envelop_value : (float) 2-sigma m threshold based on 3 months prior period
    """
    # compute the 2 sigma envelopp on the last 3 months prior to current month
    if enddate is None:
        stop_current_month = datetime.datetime.today()
    else:
        stop_current_month = enddate

    start_current_month = stop_current_month - datetime.timedelta(days=period_analysed_width)
    start_prior_period = start_current_month - datetime.timedelta(days=30 * PRIOR_PERIOD)
    stop_prior_period = start_current_month
    nb_measu_total = 0

    if df is None:
        df = load_Level2_series(satellite=satellite, start=start_prior_period, stop=stop_current_month,
                                l2f_path=l2f_path)

    log.debug(f'start_current_month : {start_current_month}', )
    log.debug(f'stop_current_month : {stop_current_month}')

    log.debug(f'prior period ; {start_prior_period} to {enddate}')
    _swh_azc_mod = df['ww3_effective_2Dcutoff_hs'].values
    log.debug('nb value WW3 eff Hs above 25 m : %s', (_swh_azc_mod > 25).sum())
    if (_swh_azc_mod > 25).sum() > 0:
        ind_bad_ww3 = np.where(_swh_azc_mod > 25)[0][0]
        log.debug(
            f'a date SAR for which ww3 is extremely too high: {df["fdatedt"][ind_bad_ww3]} -> Hs:{_swh_azc_mod[ind_bad_ww3]:1.1f}m')
    log.debug('max Hs WW3 : %s', np.nanmax(_swh_azc_mod))
    _swh_azc_s1 = df['s1_effective_hs_2Dcutoff'].values

    df['bias_swh_azc_' + wv] = xarray.DataArray(
        df['oswXA_hs_ww3spec_firstSARpartition'].values[:, 0] - df['oswXA_hs_ww3spec_firstSARpartition'].values[:, 1],
        dims=['fdatedt'], coords={'fdatedt': df['fdatedt']})

    if wv == 'wv1':
        cond_inc = (df['oswIncidenceAngle'] < 30)
    elif wv == 'wv2':
        cond_inc = (df['oswIncidenceAngle'] > 30)
    else:
        raise ValueError(f'wv value un expected : {wv}')

    polarizationcond = (df.pol == POLARIZATION.lower())
    log.debug('df.pol %s', df.pol.values)
    log.debug('polarizationcond %s', polarizationcond.values.sum())

    cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50)
    log.debug('cond_outlier_ww3_hs %s', cond_outlier_ww3_hs.values.sum())

    fini_bias = np.isfinite(df['bias_swh_azc_' + wv]) & (abs(df['bias_swh_azc_' + wv]) < 50)
    log.debug(f'fini_bias {fini_bias.values.sum()}')

    latmax_cond = (abs(df['oswLat']) < LAT_MAX)
    log.debug(f'latmax_cond {latmax_cond.values.sum()}')

    dstmax_cond = (df['dist2coastKM'] > MIN_DIST_2_COAST)
    log.debug(f'dstmax_cond {dstmax_cond.values.sum()}')

    ocean_acqui_filters = polarizationcond & latmax_cond & cond_outlier_ww3_hs \
                          & (df['oswLandFlag'] == 0) & dstmax_cond & cond_inc & fini_bias
    log.debug(f'ocean_acqui_filters {ocean_acqui_filters.values.sum()}')

    log.debug(f'start_prior_period : {start_prior_period} -> {stop_prior_period}', start_prior_period,
              stop_prior_period)

    start_prior_period64 = np.datetime64(start_prior_period)
    stop_prior_period64 = np.datetime64(stop_prior_period)
    mask_prior_period = (df['fdatedt'] >= start_prior_period64) & (df['fdatedt'] <= stop_prior_period64)
    final_mask_prior = ocean_acqui_filters & mask_prior_period
    log.debug(f'final_mask_prior {final_mask_prior.values.sum()}')
    log.debug(f'nb pts in prior period (without extra filters) : {mask_prior_period.values.sum()}')

    subset_df = df.where(final_mask_prior, drop=True)
    nb_nan = np.isnan(subset_df['bias_swh_azc_' + wv].values).sum()
    log.debug(f'nb_nan : {nb_nan}')
    log.debug(
        f'nb finite {np.isfinite(subset_df["bias_swh_azc_" + wv].values).sum()}/{len(subset_df["bias_swh_azc_" + wv])}')

    envelop_value = np.percentile(abs(subset_df['bias_swh_azc_' + wv].values), ENVELOP)
    log.debug('envelop_value : {envelop_value}')

    # compute the number of product within the envelop for current month
    start_current_month64 = np.datetime64(start_current_month)
    stop_current_month64 = np.datetime64(stop_current_month)
    current_date_cond = (df['fdatedt'] >= start_current_month64) & (df['fdatedt'] <= stop_current_month64)
    log.debug(f'current_date_cond {current_date_cond.values.sum()}')
    log.debug(f'ocean_acqui_filters {ocean_acqui_filters.values.sum()}')
    mask_current_month = ocean_acqui_filters & current_date_cond
    log.debug(f'mask_current_month {mask_current_month.values.sum()}')
    subset_current_period = df.where(mask_current_month, drop=True)
    if 'ww3_effective_2Dcutoff_hs' in subset_current_period and len(
            subset_current_period['ww3_effective_2Dcutoff_hs']) > 0:
        log.debug(f'max Hs WW3 in subset : {np.nanmax(subset_current_period["ww3_effective_2Dcutoff_hs"])}')
        log.debug(f'max Hs SAR in subset : {np.nanmax(subset_current_period["s1_effective_hs_2Dcutoff"])}')
        log.debug(f'min Hs WW3 in subset : {np.nanmin(subset_current_period["ww3_effective_2Dcutoff_hs"])}')
        log.debug(f'min Hs SAR in subset : {np.nanmin(subset_current_period["s1_effective_hs_2Dcutoff"])}')
        log.debug(f'nb pts current month : {len(subset_current_period["fdatedt"])}')
        nb_measu_total = len(subset_current_period['fdatedt'])
        log.debug(f'bias : {subset_current_period["bias_swh_azc_" + wv].values}')

        # definition proposee par Hajduch le 10dec2021 screenshot a lappuit (je ne suis pas convaincu pas l introduction du biais dans le calcul de levenveloppe car le KPI sera dautant plus elever que le biais sera fort (cest linverse qui est cherche)
        bias_minus_2sigma = abs(subset_current_period['bias_swh_azc_' + wv].mean().values - envelop_value)
        bias_plus_2sigma = abs(subset_current_period['bias_swh_azc_' + wv].mean().values + envelop_value)
        log.info(f'bias_plus_2sigma {bias_plus_2sigma}')
        T = np.max([bias_minus_2sigma, bias_plus_2sigma])
        log.info(f'T : {T.shape} {T}')
        nb_measu_inside_envelop = (abs(subset_current_period['bias_swh_azc_' + wv]) < T).sum().values
        std = np.nanstd(subset_current_period['bias_swh_azc_' + wv])
        mean_bias = np.mean(subset_current_period['bias_swh_azc_' + wv]).values
        log.debug(f'nb_measu_inside_envelop : {nb_measu_inside_envelop}')
        kpi_value = 100. * nb_measu_inside_envelop / nb_measu_total
        log.debug(f'kpi_value : {kpi_value}')
        if dev:
            from matplotlib import pyplot as plt
            plt.figure()
            binz = np.arange(0, 10, 0.1)
            hh, _ = np.histogram(subset_current_period['ww3_effective_2Dcutoff_hs'], binz)
            plt.plot(binz[0:-1], hh, label='WWIII %s' % len(subset_current_period['ww3_effective_2Dcutoff_hs']))
            hh, _ = np.histogram(subset_current_period['s1_effective_hs_2Dcutoff'], binz)
            plt.plot(binz[0:-1], hh, label='SAR %s' % len(subset_current_period['ww3_effective_2Dcutoff_hs']))
            plt.grid(True)
            plt.legend()
            plt.xlabel('Hs (m)')
            output = '/home1/scratch/agrouaze/test_histo_kpi_1d.png'
            plt.savefig(output)
            log.debug('png test : %s', output)
            # plt.show()
    else:
        mean_bias = np.nan
        kpi_value = np.nan
        std = np.nan
        log.debug('no data for period %s to %s', start_current_month, stop_current_month)
        log.debug('subset_current_period %s', subset_current_period)

    item = KPI1DItem(kpi_value, start_current_month, stop_current_month, envelop_value, nb_measu_total, mean_bias, std)

    if output:
        output_file = os.path.join(output, f'kpi_output_{sat}_{wv}_{enddate if enddate else ""}.txt')
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

    kpi = compute_kpi_1d(**params)
    log.info('#' * 10)
    log.info(
        f'kpi_v {params["satellite"]} {params["wv"]} : {kpi.value} (envelop {ENVELOP}-sigma value: {kpi.envelop_value} dB)')
    log.info(f'nb pts used: {kpi.count}')
    log.info('#' * 10)
    log.info(f'start_cur_month : {kpi.start}, stop_cur_month : {kpi.stop}')

    log.info(f'done in {(time.time() - t0) / 60.:1.3f} min')
    log.info(f'peak memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.} Mbytes')


log.info("over")

if __name__ == '__main__':
    main()
