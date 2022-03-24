# Kpi Mpc

libs to compute 2 KPIs  for MPC Sentinel-1 project:
  - a) WV VV effective Hs bias (wrt WW3)
  - b) WV VV NRCS (denoised) bias (wrt CMOD-5n)

## Authors:
    IFREMER LOPS
    antoine.grouazel@ifremer.fr
    CLS
    tlechauve@groupcls.com

## Build & Installation

### (optionally) Build virtual environment with conda
```bash
conda create -n kpi_conda_env python=3.9
conda activate kpi_conda_env
conda install numpy scipy matplotlib xarray netCDF4 ipykernel 
```

###Get sources
```bash
git clone https://github.com/umr-lops/kpi_mpc.git`
```

### Installation
```bash
cd kpi_mpc
python setup.py install 
```

### or with Pip

```bash
cd kpi_mpc
pip install -r requirements.txt
pip install . (use `-e` in developer mode)
```

## Usage

### KPI-1B (NRCS SLC)
KPI-1b needs a netCDF file containing the differences of NRCS denoised per SAFE for all the WV since 2015 ( S1%_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc)

```bash
usage: compute_kpi_1b [-h] [-v] [-f] [-c CONFIG] [--enddate ENDDATE]
                      [-o OUTPUT] [--coastline-netcdf COASTLINE_NETCDF]
                      [--inputs-path-pattern INPUTS_PATH_PATTERN]
                      {S1A,S1B} {wv1,wv2}

Compute KPI-1B and save results as textfile

Use configuration file (--config) to avoid a long list of parameters. Example; kpi1b.ini

   [DEFAULT]
    output=/tmp/output
    coastline_netcdf=/tmp/NASA_tiff_distance_to_coast_converted_v2.nc
    input_path_pattern=/tmp/%%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc

! Careful inline arguments in command will overwrite its value in configuration file.

positional arguments:
  {S1A,S1B}             S-1 unit choice
  {wv1,wv2}             WV incidence angle choice

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose
  -f, --overwrite       overwrite the existing outputs [default=False]
  -c CONFIG, --config CONFIG
                        Path to the configuration file.
  --enddate ENDDATE     end of the 1 month period analysed
  -o OUTPUT, --output OUTPUT
                        Path output will be stored
  --coastline-netcdf COASTLINE_NETCDF
                        Coastline NetCDF
  --inputs-path-pattern INPUTS_PATH_PATTERN
                        Path to input files as pattern, the satellite unit
                        will bepassed as parameter for this pattern. Use %s
                        where the unitshould be placed in the filename. Use
                        %%s in the configfile
```

### KPI-1d (Hs OCN)
KPI-1d is designed to run using Sentinel-1 WV IFREMER L2F daily aggregated nc files.

```bash
usage: compute_kpi_1d [-h] [-v] [-f] [-c CONFIG] [--enddate ENDDATE]
                      [-o OUTPUT] [--l2f L2F]
                      {S1A,S1B} {wv1,wv2}

Compute KPI-1D and save results as textfile

Use configuration file (--config) to avoid a long list of parameters. Example; kpi1b.ini

   [DEFAULT]
    output=/tmp/output
    l2f=/tmp

! Careful inline arguments in command will overwrite its value in configuration file.

positional arguments:
  {S1A,S1B}             S-1 unit choice
  {wv1,wv2}             WV incidence angle choice

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose
  -f, --overwrite       overwrite the existing outputs [default=False]
  -c CONFIG, --config CONFIG
                        Path to the configuration file.
  --enddate ENDDATE     end of the 1 month period analysed
  -o OUTPUT, --output OUTPUT
                        Path output will be stored
  --l2f L2F             Path to directory for alternative L2F
```
