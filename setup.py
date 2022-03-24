from setuptools import setup, find_packages

setup(
    name='kpi-mpc',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=[],
    url='https://github.com/tlechauveCLS/kpi_mpc',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        'xarray>=0.14.0',
        'netCDF4>=1.5.1.2',
        'numpy>=1.17.3',
        'scipy>=1.3.1',
    ],
    entry_points={
        'console_scripts': {
            'compute_kpi_1b=mpc.kpi1b.compute_kpi_1b:main'
        }
    },
    license='MIT',
    author='Antoine Grouazel, Thomas Lechauve',
    author_email='Antoine.grouazel@ifremer.fr, tlechauve@groupcls.com',
    description='libraries to compute Key Performance Indicators for Mission Performance Center (Sentinel-1 SAR mission)',
    namespace_packages=['mpc']
)
