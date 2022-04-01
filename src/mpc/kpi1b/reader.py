import glob
import logging
import time
from datetime import datetime

import netCDF4
import numpy as np
import os
import xarray

from mpc.kpi1b.compute_wind_azimuth import method_wind_azi_range
from mpc.kpi1b.gmf_cmod5n import GMFCmod5n
# first test to look at the content of the daily files SAFE containing the noise and denoised sigma0
from mpc.kpi1b.parsers import SATELLITE_LIST

""" 
read_aggregated_calbration_SLC_WV_level_netcdf_file_for_nrcs_investigations
"""


def read_fat_calib_nc(input_files, coastline_netcdf, satellite_list=None):
    """ read the fat netcdf files for NRCS investigations build from ocean_wv_calibration_huimin_method.py
    :return:
    """

    if not satellite_list:
        satellite_list = SATELLITE_LIST

    df_slc_sat = {}
    for sat in satellite_list:
        pattern = input_files % sat
        logging.info('pattern %s', pattern)
        try:
            output_file_calibration = glob.glob(pattern)[0]
        except IndexError:
            raise FileNotFoundError(pattern)
        logging.info('found %s', output_file_calibration)
        # logging.debug('is same file %s',output_file_calibration == testf)
        if os.path.exists(output_file_calibration):
            logging.info('read file %s', output_file_calibration)
            nctest = netCDF4.Dataset(output_file_calibration)
            logging.debug('var kurt_sar %s', 'kurt_sar' in nctest.variables.keys())
            logging.debug('var skew_sar %s', 'skew_sar' in nctest.variables.keys())
            nctest.close()
            xd = xarray.open_dataset(output_file_calibration)
            df_slc = xd.to_dataframe()
            df_slc['time'] = xd['time'].values
            df_slc.index = df_slc['time']
            df_slc.drop_duplicates(subset=['time'], inplace=True)
            # drop NaN latitudes
            logging.info('before lat filter %s', len(df_slc))
            df_slc = df_slc.dropna(subset=['_lat_sar'])
            df_slc = df_slc[(df_slc['_lat_sar'] > -90) & (df_slc['_lat_sar'] < 90)]
            logging.info('after lat filter %s finite %s %s', len(df_slc), np.isfinite(df_slc['_lat_sar']).sum(),
                         np.isfinite(df_slc['_lon_sar']).sum())
            # print(df_slc)
            df_slc.dropna(subset=['time'], inplace=True)
            df_slc['_sig_sar_db'] = 10. * np.log10(df_slc['_sig_sar'])
            df_slc['sigma0_denoised_db'] = 10. * np.log10(df_slc['sigma0_denoised'])

            t0 = time.time()
            dst = get_distance_to_coast_vecto(df_slc['_lon_sar'].values,
                                              df_slc['_lat_sar'].values,
                                              coastline_netcdf)
            logging.info('elapsed time to have the data reade %1.1f seconds' % (time.time() - t0))
            df_slc['distance2coast'] = dst
            # add the windazi
            test_trackangle = (df_slc['_tra_sar']) % 360
            azi_sar_test = method_wind_azi_range(df_slc['_zon_coloc_ecmwf'],
                                                 df_slc['_mer_coloc_ecmwf'],
                                                 test_trackangle)
            df_slc['_wind_azimuth_ecmwf'] = azi_sar_test

            gmf = GMFCmod5n()
            sigma0_cmod5n = gmf._getNRCS(df_slc['_inc_sar'].values, df_slc['_spd_coloc_ecmwf'].values, azi_sar_test)
            df_slc['tmp_gmf_cmod5n_nrcs'] = sigma0_cmod5n
            df_slc['tmp_gmf_cmod5n_nrcs_db'] = 10. * np.log10(sigma0_cmod5n)

            # fix ecmwf wind directions
            _dirtest2 = np.mod(np.degrees(np.arctan2(df_slc['_zon_coloc_ecmwf'], df_slc['_mer_coloc_ecmwf'])), 360.)
            df_slc['_dir_coloc_ecmwf'] = _dirtest2
            # define nosie in db
            df_slc['noise_db'] = 10.0 * np.log10(df_slc['noise'])
            # save the dataframe enriched in dict
            logging.info('%s df enriched ok', sat)
            df_slc_sat[sat] = df_slc

        else:
            print('no %s' % output_file_calibration)
    return df_slc_sat


def latlon2ij(lat, lon, shape2D, llbox):
    """
    convert lat,lon into i,j index
    args:
        lat (float or 1D nd.array):
        lon (float or 1D nd.array):
        shape2D (tuple): (10,20) for instance
        llbox (tuple): latmin, lonmin, latmax,lonmax
    """
    logging.debug('input lat latlon2ij | %s', lat)
    latmin, lonmin, latmax, lonmax = llbox
    if isinstance(lat, float) or isinstance(lat, int):
        lat = np.array([lat])
    if isinstance(lon, float) or isinstance(lon, int):
        lon = np.array([lon])
    dlon = lonmax - lonmin
    dlat = latmax - latmin
    logging.debug('dlon = %s', dlon)
    logging.debug('dlat = %s', dlat)
    logging.debug('shape2D = %s', shape2D)
    logging.debug('lat type %s %s', type(lat), lat)
    logging.debug('lat range %s %s', lat.min(), lat.max())
    logging.debug('dlat %s shapz %s', dlat, shape2D)
    logging.debug('itest %s', np.floor((lat - latmin) * shape2D[0] / dlat))
    i = np.floor((lat - latmin) * shape2D[0] / dlat).astype(
        int)  # changed May 2019 after founding a bug with B. Coatanea where indices can reach the maximum value of the shape... (agrouaze)
    j = np.floor((lon - lonmin) * shape2D[1] / dlon).astype(int)

    return i, j


def get_latlon_coasts(netcdf_path):
    """ Get latitude/longitude and distance to coast from the netcdf variables

    :param netcdf_path: path to the netcdf
    :return: list (lat, lon, dist)
    """
    nc = netCDF4.Dataset(netcdf_path)
    dist = nc.variables['distance_to_coast'][:]
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    nc.close()

    return lat, lon, dist


def get_distance_to_coast_vecto(lons, lats, coastline_netcdf):
    """

    :param lons: list of longitudes
    :param lats: list of latitude
    :param netcdf_path:
    :return:
    """

    coast_lat, coast_lon, coast_dist = get_latlon_coasts(coastline_netcdf)

    llbox = (coast_lat[0], coast_lon[0], coast_lat[-1], coast_lon[-1])
    indlat, indlon = latlon2ij(lats, lons, np.shape(coast_dist), llbox)
    indlat[(indlat >= coast_dist.shape[0])] = coast_dist.shape[0] - 1
    indlon[(indlon >= coast_dist.shape[1])] = coast_dist.shape[1] - 1
    dsts = coast_dist[indlat, indlon]
    diff_lon = lons - coast_lon[indlon]
    diff_lat = lats - coast_lat[indlat]

    return dsts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_date = datetime(2015, 1, 1)
    stop_date = datetime(2019, 12, 1)
    stop_date = datetime(2020, 1, 9)

    # stop_date= datetime(2019,12,31)
    # start_date = datetime(2019,2,1)
    # stop_date= datetime(2019,3,20)
    # start_date = datetime(2016,1,1)
    # start_date = datetime(2019,8,27)
    # stop_date= datetime(2018,2,28)
    # start_date = datetime(2019,1,1)
    # stop_date= datetime(2019,10,10)
    level = 'L1'
    satellite = 'S1A'
    subdir = 'v11'  # ecmwf0125 with corrcted wind direction computation
    subdir = 'v12'  # correction noise sur IPF2.91 + dlon et dlon coloc ecmwf
    subdir = 'v14'  # add roughness classification
    # subdir = 'v15' # kurt + skew encmwf horaire
    subdir = 'v16'  # new noise vectors cst correction from Pauline
    if subdir == 'v15':
        start_date = datetime(2019, 8, 21)  # v15
        stop_date = datetime(2020, 2, 20)
    if subdir == 'v16':
        stop_date = datetime(2020, 3, 27)

    # sta_str = start_date.strftime('%Y%m%d')
    # sto_str = stop_date.strftime('%Y%m%d')
    df_slc_sat = read_fat_calib_nc(['S1A'])
