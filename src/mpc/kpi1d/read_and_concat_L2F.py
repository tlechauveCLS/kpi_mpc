# encoding -utf8-
"""
read and concatenate daily files variables into a xarray Datatset
"""
import datetime
import glob
import logging

import numpy as np
import os
import xarray

from mpc.common import SATELLITE_LIST

log = logging.getLogger('mpc.compute_kpi_1d.read_concat_l2f')


def preprocess_L2F(ds, variables=None, add_ecmwf_wind=True):
    """
    
    :param ds:
    :param variables:
    :return:
    """
    ds = ds.sortby('fdatedt')
    if variables is not None:
        ds = ds[variables]
        for kk in ds.keys():
            log.debug('test kk : %s', kk)
            if 'dataset' in ds[kk].dims:
                log.debug('split %s in S1 + WW3 vars')
                ds[kk + '_ww3'] = xarray.DataArray(ds[kk][:, 1].values,
                                                   dims=['fdatedt'])  # checked in partition_wv_xassignement.py
                ds[kk + '_s1'] = xarray.DataArray(ds[kk][:, 0].values, dims=['fdatedt'])
    if 'ecmwf_wind_speed' not in ds and add_ecmwf_wind:
        tmpval = np.sqrt(ds['ecmwf0125_uwind'].values ** 2. + ds['ecmwf0125_vwind'].values ** 2.)
        ds['ecmwf_wind_speed'] = xarray.DataArray(tmpval, dims=['fdatedt'])
    if 'pol' in ds:
        ds['pol'] = ds['pol'].astype(str)

    ds = ds.assign_coords({'dataset': ['sar', 'ww3']})  # to fix a S1B 20210223 file...
    return ds


def read_L2F_with_xarray(start, stop, l2f_path, satellites=None, variables=None, add_ecmwf_wind=True):
    """

    :param start:
    :param stop:
    :param satellites:
    :param variables:
    :param l2f_path:
    :param add_ecmwf_wind :bool
    :return:
    """
    if not satellites:
        satellites = SATELLITE_LIST

    if isinstance(start, datetime.date):
        start = datetime.datetime(start.year, start.month, start.day)
    if isinstance(stop, datetime.date):
        stop = datetime.datetime(stop.year, stop.month, stop.day)
    log.info('Sentinel-1 L2F lecture between %s and %s', start, stop)
    ds_dict_sat = {}
    for satr in satellites:
        ds_dict_sat[satr] = {}
    for sensor in satellites:  # pas de S1B pour le moment car pas de fichier avec les varaibles cross assigned 28 janvier 2020
        if start.year == stop.year:
            pat = os.path.join(l2f_path, start.strftime('%Y'), '*', sensor + '*nc')
        else:
            pat = os.path.join(l2f_path, '*', '*', sensor + '*nc')
        log.info('pattern to search for L2F =%s', pat)
        listnc0 = sorted(glob.glob(pat))[::-1]
        listnc = []
        dates = []
        for ff in listnc0:
            datdt = datetime.datetime.strptime(os.path.basename(ff).split('_')[5], '%Y%m%d')
            if datdt >= start and datdt <= stop and datdt not in dates:
                listnc.append(ff)
                dates.append(datdt)
        log.info(f"nb files found without date filter = {len(listnc0)}", )
        log.info(f"nb files found with date filter = {len(listnc)}")
        for kk in listnc:
            log.debug(kk)
        if len(listnc) > 0:
            tmpds = xarray.open_mfdataset(listnc, preprocess=lambda ds: preprocess_L2F(ds, variables, add_ecmwf_wind),
                                          combine='by_coords')  # , concat_dim='fdatedt')
            ds_dict_sat[sensor] = tmpds

    return ds_dict_sat


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='read and concat')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    start = datetime.datetime(2021, 7, 1)
    stop = datetime.datetime(2021, 8, 2)
    start = datetime.datetime(2021, 7, 24)
    stop = datetime.datetime(2021, 8, 9)
    print('start', start)
    logging.info('%s %s', start, stop)
    test_vars = ['oswWindSpeed', 'oswLon', 'oswLat', 'fdatedt', 'oswHeading', 'wind_speed_model', 'azimuth_angle',
                 'wind_speed_model_u',
                 'wind_speed_model_v', 'ecmwf0125_uwind', 'ecmwf0125_vwind', 'rvlNrcsGridmean', 'class_1']
    test_vars = None
    sat = ['S1A', 'S1B']
    ds_sat = read_L2F_with_xarray(start, stop, satellites=['S1A', 'S1B'], variables=None, alternative_L2F_path=None,
                                  add_ecmwf_wind=True)
    logging.info(ds_sat.keys())
    logging.info('var = %s', ds_sat[sat[0]].keys())
    logging.info('nb dates : %s', len(ds_sat[sat[0]]['fdatedt']))
    logging.info('over')
    print('over')
