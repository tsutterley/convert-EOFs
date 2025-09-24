# API Reference

## `convert_atl11_atl15_heights.py`

```
usage: convert_atl11_atl15_heights.py [-h] [--mask MASK [MASK ...]] [--atl15 ATL15 [ATL15 ...]] [--lmax LMAX] [--mmax MMAX] [--radius RADIUS] [--verbose] [--mode MODE] infile

Convert ATL11/ATL15 heights to spherical harmonics and then back to the spatial domain after spectral processing

positional arguments:
  infile                Input ATL11/ATL15 data file

options:
  -h, --help            show this help message and exit
  --mask MASK [MASK ...]
                        netCDF4 masks file for reducing to regions
  --atl15, -A ATL15 [ATL15 ...]
                        ATL15 file to use for areas
  --lmax, -l LMAX       Maximum spherical harmonic degree
  --mmax, -m MMAX       Maximum spherical harmonic order
  --radius, -R RADIUS   Gaussian smoothing radius (km)
  --verbose, -V         Verbose output of processing run
  --mode, -M MODE       Permission mode of directories and files
```

## `convert_basin_masks.py`

```
usage: convert_basin_masks.py [-h] [--atl15 ATL15 [ATL15 ...]] [--lmax LMAX] [--mmax MMAX] [--radius RADIUS] [--sigma SIGMA] [--tolerance TOLERANCE] [--verbose] [--mode MODE] infile

Convert drainage basin masks to spherical harmonics and then back to the spatial domain after spectral processing

positional arguments:
  infile                Input drainage basin file

options:
  -h, --help            show this help message and exit
  --atl15, -A ATL15 [ATL15 ...]
                        ATL15 file to use for areas
  --lmax, -l LMAX       Maximum spherical harmonic degree
  --mmax, -m MMAX       Maximum spherical harmonic order
  --radius, -R RADIUS   Gaussian smoothing radius (km)
  --sigma, -S SIGMA     Standard deviation for Gaussian kernel
  --tolerance, -T TOLERANCE
                        Tolerance to set as valid mask
  --verbose, -V         Verbose output of processing run
  --mode, -M MODE       Permission mode of directories and files
```

## `convert_elastic_heights.py`

```
usage: convert_elastic_heights.py [-h] [--mask MASK [MASK ...]] [--atl15 ATL15 [ATL15 ...]] [--lmax LMAX] [--mmax MMAX] [--radius RADIUS] [--verbose] [--mode MODE] infile

Convert heights to spherical harmonics and then back to the spatial domain after spectral processing

positional arguments:
  infile                Input data file

options:
  -h, --help            show this help message and exit
  --mask MASK [MASK ...]
                        netCDF4 masks file for reducing to regions
  --atl15, -A ATL15 [ATL15 ...]
                        ATL15 file to use for areas
  --lmax, -l LMAX       Maximum spherical harmonic degree
  --mmax, -m MMAX       Maximum spherical harmonic order
  --radius, -R RADIUS   Gaussian smoothing radius (km)
  --verbose, -V         Verbose output of processing run
  --mode, -M MODE       Permission mode of directories and files
```

## `python convert_height_eofs.py`

```
usage: convert_height_eofs.py [-h] [--mask MASK [MASK ...]] [--atl15 ATL15 [ATL15 ...]] [--lmax LMAX] [--mmax MMAX] [--radius RADIUS] [--verbose] [--mode MODE] infile

Convert Height EOFs to spherical harmonics and then back to the spatial domain after spectral processing

positional arguments:
  infile                Input EOF data file

options:
  -h, --help            show this help message and exit
  --mask MASK [MASK ...]
                        netCDF4 masks file for reducing to regions
  --atl15, -A ATL15 [ATL15 ...]
                        ATL15 file to use for areas
  --lmax, -l LMAX       Maximum spherical harmonic degree
  --mmax, -m MMAX       Maximum spherical harmonic order
  --radius, -R RADIUS   Gaussian smoothing radius (km)
  --verbose, -V         Verbose output of processing run
  --mode, -M MODE       Permission mode of directories and files
```

## `convert_smb_fac_eofs.py`

```
usage: convert_smb_fac_eofs.py [-h] [--mask MASK [MASK ...]] [--atl15 ATL15 [ATL15 ...]] [--lmax LMAX] [--mmax MMAX] [--radius RADIUS] [--verbose] [--mode MODE] infile

Convert SMB and FAC EOFs to spherical harmonics and then back to the spatial domain after spectral processing

positional arguments:
  infile                Input EOF data file

options:
  -h, --help            show this help message and exit
  --mask MASK [MASK ...]
                        netCDF4 masks file for reducing to regions
  --atl15, -A ATL15 [ATL15 ...]
                        ATL15 file to use for areas
  --lmax, -l LMAX       Maximum spherical harmonic degree
  --mmax, -m MMAX       Maximum spherical harmonic order
  --radius, -R RADIUS   Gaussian smoothing radius (km)
  --verbose, -V         Verbose output of processing run
  --mode, -M MODE       Permission mode of directories and files
```