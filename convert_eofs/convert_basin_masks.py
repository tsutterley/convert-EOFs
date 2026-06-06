#!/usr/bin/env python
"""
convert_basin_masks.py
Written by Tyler Sutterley (03/2023)
Read drainage basin masks and convert to spherical harmonics
Convert back into the spatial domain after truncation and smoothing
"""

from __future__ import print_function

import sys
import re
import copy
import time
import pyproj
import logging
import pathlib
import netCDF4
import argparse
import warnings
import numpy as np
import scipy.ndimage
import gravity_toolkit as gravtk
import model_harmonics as mdlhmc
from convert_eofs.ATL15 import mosaic_ATL15

# ignore pyproj and divide by zero warnings
warnings.filterwarnings("ignore")


# PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if REGION in ("ais",):
        projection_flag = "EPSG:3031"
    elif REGION in ("gris",):
        projection_flag = "EPSG:3413"
    return projection_flag


def convert_basin_masks(
    INPUT_FILE,
    LMAX,
    MMAX=None,
    RAD=0,
    ATL15=None,
    BUFFER=800e3,
    SIGMA=0,
    TOLERANCE=1,
    MODE=0o775,
):
    # verify input file
    INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Mask file not found in file system")
    # read mask file and extract variables
    regex_pattern = r"(.*?)_(sep_)?(ais|gris)(.*?).nc$"
    PREFIX, SEP, REGION, AUX = re.findall(regex_pattern, INPUT_FILE.name).pop()
    logging.info(str(INPUT_FILE))
    logging.debug(f"Prefix:{PREFIX}")
    logging.debug(f"Region:{SEP}{REGION}")
    fileID = netCDF4.Dataset(INPUT_FILE, mode="r")
    fd = {}
    for key, val in fileID.variables.items():
        fd[key] = val[:]
    # invalid data value
    fv = np.float64(fileID.variables["mask"]._FillValue)
    # input variable units
    variable_units = fileID.variables["mask"].units
    # fix Greenland ATL15 coordinates
    dx = np.abs(fd["x"][1] - fd["x"][0])
    if (REGION.lower() == "gris") and (dx != 10e3):
        fd["x"] = fd["x"][0] + 10e3 * np.arange(len(fd["x"]))
    # calculate grid areas (assume fully ice covered)
    dx = np.abs(fd["x"][1] - fd["x"][0])
    dy = np.abs(fd["y"][1] - fd["y"][0])
    # verify input data shape
    if np.ndim(fd["mask"]) == 2:
        fd["mask"] = np.atleast_3d(fd["mask"]).transpose(2, 0, 1)
    # input shape of input data
    nband, ny, nx = np.shape(fd["mask"])
    shape = (ny, nx)
    output_shape = (ny + int(2 * BUFFER // dy), nx + int(2 * BUFFER // dx))
    indexing = "xy"
    logging.debug(f"Shape: {shape}")
    logging.debug(f"Output shape: {output_shape}")
    # create band variable
    fd["band"] = np.arange(1, nband + 1)
    # extract x and y coordinate arrays
    xg, yg = np.meshgrid(fd["x"], fd["y"], indexing=indexing)
    # close the netCDF4 file
    fileID.close()

    # pyproj transformer for converting to input coordinates (EPSG)
    MODEL_EPSG = set_projection(REGION)
    crs1 = pyproj.CRS.from_string("EPSG:4326")
    crs2 = pyproj.CRS.from_string(MODEL_EPSG)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    # get reference parameters for ellipsoid
    ellipsoid_params = mdlhmc.datum(ellipsoid="WGS84")
    # semi-major axis of ellipsoid [m]
    a_axis = ellipsoid_params.a_axis
    # flattening of the ellipsoid
    flat = ellipsoid_params.flat
    # Average Radius of the Earth with equal surface area [m]
    rad_e = ellipsoid_params.rad_e

    # convert projection from model coordinates
    modellon, modellat = transformer.transform(xg, yg, direction=direction)
    # convert latitudes to geocentric latitudes
    latitude_geocentric = mdlhmc.spatial.geocentric_latitude(
        modellon, modellat, a_axis=a_axis, flat=flat
    )
    # polar stereographic standard parallel (latitude of true scale)
    reference_latitude = crs2.to_dict().pop("lat_ts")

    # output as buffered grid
    output = dict(band=np.copy(fd["band"]))
    xmin, xmax = (fd["x"].min() - BUFFER, fd["x"].max() + BUFFER)
    ymin, ymax = (fd["y"].min() - BUFFER, fd["y"].max() + BUFFER)
    output["x"] = np.arange(xmin, xmax + dx, dx)
    output["y"] = np.arange(ymin, ymax + dy, dy)
    xout, yout = np.meshgrid(output["x"], output["y"], indexing=indexing)
    # convert projection from model coordinates
    bufferlon, bufferlat = transformer.transform(
        xout, yout, direction=direction
    )
    # convert latitudes to geocentric latitudes
    buffer_latitude_geocentric = mdlhmc.spatial.geocentric_latitude(
        bufferlon, bufferlat, a_axis=a_axis, flat=flat
    )

    # fix pole hole
    ii, jj = np.nonzero(latitude_geocentric <= -87.5)
    for band in range(nband):
        fd["mask"][band, ii, jj] = 1.0

    # Gaussian filter the mask to create an initial smoothed version
    if SIGMA > 0:
        # convert nan values to 0
        for band in range(nband):
            temp = np.nan_to_num(fd["mask"][band, :, :].filled(0.0), nan=0.0)
            # gaussian filter image
            temp = scipy.ndimage.gaussian_filter(
                temp.astype(np.float64), SIGMA, mode="constant", cval=0
            )
            ii, jj = np.nonzero(temp > TOLERANCE)
            fd["mask"][band, ii, jj] = 1.0

    # reduce latitude and longitude to valid and masked points
    indx, indy = np.nonzero(np.any((fd["mask"] != fv), axis=0))
    lon, lat = (modellon[indx, indy], latitude_geocentric[indx, indy])
    # input area grids (scaled for polar stereographic distortion)
    if ATL15:
        # use ATL15 ice area for scaling
        mosaic = mosaic_ATL15(ATL15)
        fd["area"] = np.max(mosaic["ice_area"].filled(fill_value=0), axis=0)
        logging.debug("Area shape: {0}".format(fd["area"].shape))
    else:
        # scaled areas for polar stereographic distortion
        ps_scale = mdlhmc.spatial.scale_factors(
            modellat[indx, indy],
            flat=flat,
            reference_latitude=reference_latitude,
        )
        fd["area"] = np.zeros((shape))
        fd["area"][indx, indy] = ps_scale * dx * dy
    # areas in terms of solid angle (steradians)
    scaling_factors = fd["area"][indx, indy] / (rad_e**2)
    # degree-dependent spherical harmonic units (4-pi normalized)
    UNITS = np.ones((LMAX + 1)) / (4.0 * np.pi)

    # upper bound of spherical harmonic orders (default = LMAX)
    MMAX = np.copy(LMAX) if not MMAX else MMAX
    # output string for both LMAX == MMAX and LMAX != MMAX cases
    order_str = f"M{MMAX:d}" if (MMAX != LMAX) else ""
    # Calculating the Gaussian smoothing for radius RAD
    gw_str = f"_r{RAD:0.0f}km" if (RAD != 0) else ""

    # attributes for output files
    attributes = {}
    attributes["units"] = copy.copy(variable_units)
    attributes["reference"] = f"Output from {pathlib.Path(sys.argv[0]).name}"
    # for each variable
    for var in ["mask"]:
        # allocate for output spherical harmonics
        Ylms = mask(lmax=LMAX, mmax=MMAX)
        Ylms.clm = np.zeros((LMAX + 1, MMAX + 1, nband))
        Ylms.slm = np.zeros((LMAX + 1, MMAX + 1, nband))
        Ylms.band = np.copy(fd["band"])

        # output spatial
        output[var] = np.ma.zeros((nband, *output_shape), fill_value=fv)
        # for each band
        for n in range(nband):
            # reduce data to band and scale areas
            SCALED = np.nan_to_num(scaling_factors * fd[var][n, indx, indy], 0)
            # convert scaled values to spherical harmonics
            # use custom UNITS to keep as inputs but use 4-pi norm
            YLMS = gravtk.gen_point_load(
                SCALED, lon, lat, LMAX=LMAX, MMAX=MMAX, UNITS=UNITS
            )
            # copy spherical harmonics for band
            Ylms.clm[:, :, n] = YLMS.clm[:, :].copy()
            Ylms.slm[:, :, n] = YLMS.slm[:, :].copy()
            # convert spherical harmonics to spatial domain
            # using buffered grid coordinates
            # use custom UNITS to keep as inputs
            spatial = gravtk.clenshaw_summation(
                Ylms.clm[:, :, n],
                Ylms.slm[:, :, n],
                bufferlon.flatten(),
                buffer_latitude_geocentric.flatten(),
                RAD=RAD,
                LMAX=LMAX,
                UNITS=np.ones((LMAX + 1)),
            )
            # reshape to output and save for band
            output[var][n, :, :] = spatial.reshape(output_shape)

        # output spherical harmonic data file for variable
        FILE = f"{PREFIX}_{SEP}{REGION}_CLM_L{LMAX:d}{order_str}.nc"
        CLM_FILE = INPUT_FILE.with_name(FILE)
        Ylms.to_netCDF4(CLM_FILE, date=False, **attributes)
        # change the permissions mode of the output file to MODE
        CLM_FILE.chmod(mode=MODE)

    # output smoothed mask file
    FILE = f"{PREFIX}_{SEP}{REGION}{AUX}_L{LMAX:d}{order_str}{gw_str}.nc"
    OUTPUT_FILE = INPUT_FILE.with_name(FILE)
    output_to_netCDF4(OUTPUT_FILE, output, region=REGION)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)


# PURPOSE: output gridded data to netCDF4
def output_to_netCDF4(output_file, output, **kwargs):
    # set default keyword arguments
    kwargs.setdefault("region", "gris")

    # opening NetCDF file for writing
    logging.info(str(output_file))
    fileID = netCDF4.Dataset(output_file, "w", format="NETCDF4")

    # output shape of data
    nband, ny, nx = np.shape(output["mask"])
    dims = (
        "band",
        "y",
        "x",
    )

    # Defining the NetCDF dimensions
    fileID.createDimension("x", nx)
    fileID.createDimension("y", ny)
    fileID.createDimension("band", nband)

    # python dictionary with netCDF4 variables
    nc = {}
    # defining the NetCDF variables
    nc["x"] = fileID.createVariable("x", output["x"].dtype, ("x",))
    nc["y"] = fileID.createVariable("y", output["y"].dtype, ("y",))
    nc["band"] = fileID.createVariable("band", output["band"].dtype, ("band",))
    # for each output variable
    for v in ["mask"]:
        nc[v] = fileID.createVariable(
            v, output[v].dtype, dims, fill_value=output[v].fill_value, zlib=True
        )

    # filling NetCDF variables
    for key, val in output.items():
        nc[key][:] = val.copy()

    # create variable and attributes for projection
    if kwargs["region"] in ("gris",):
        crs = fileID.createVariable("Polar_Stereographic", np.byte, ())
        crs.standard_name = "Polar_Stereographic"
        crs.grid_mapping_name = "polar_stereographic"
        crs.straight_vertical_longitude_from_pole = -45.0
        crs.latitude_of_projection_origin = 90.0
        crs.standard_parallel = 70.0
        crs.scale_factor_at_projection_origin = 1.0
        crs.false_easting = 0.0
        crs.false_northing = 0.0
        crs.semi_major_axis = 6378.137
        crs.semi_minor_axis = 6356.752
        crs.inverse_flattening = 298.257223563
        crs.spatial_epsg = "3413"
    elif kwargs["region"] in ("ais",):
        crs = fileID.createVariable("Polar_Stereographic", np.byte, ())
        crs.standard_name = "Polar_Stereographic"
        crs.grid_mapping_name = "polar_stereographic"
        crs.straight_vertical_longitude_from_pole = 0.0
        crs.latitude_of_projection_origin = -90.0
        crs.standard_parallel = -71.0
        crs.scale_factor_at_projection_origin = 1.0
        crs.false_easting = 0.0
        crs.false_northing = 0.0
        crs.semi_major_axis = 6378.137
        crs.semi_minor_axis = 6356.752
        crs.inverse_flattening = 298.257223563
        crs.spatial_epsg = "3031"

    # Defining attributes for x and y coordinates
    nc["x"].long_name = "Easting"
    nc["x"].standard_name = "projection_x_coordinate"
    nc["x"].grid_mapping = "Polar_Stereographic"
    nc["x"].units = "meters"
    nc["y"].long_name = "Northing"
    nc["y"].standard_name = "projection_y_coordinate"
    nc["y"].grid_mapping = "Polar_Stereographic"
    nc["y"].units = "meters"
    # Defining attributes for variables
    for v in ["mask"]:
        # set variable attributes
        nc[v].setncattr("units", "1")
        nc[v].setncattr("standard_name", v)
        # set grid mapping attribute
        nc[v].setncattr("grid_mapping", "Polar_Stereographic")

    # global attributes of output netCDF4 file
    fileID.title = "Basin mask for joint solution"
    fileID.date_created = time.strftime("%Y-%m-%d", time.localtime())
    fileID.reference = (
        "Medley, B., Neumann, T. A., Zwally, H. J., "
        "Smith, B. E., and Stevens, C. M.: Simulations of Firn Processes "
        "over the Greenland and Antarctic Ice Sheets: 1980--2021, "
        "The Cryosphere, https://doi.org/10.5194/tc-16-3971-2022, 2022."
    )
    fileID.institution = "NASA Goddard Space Flight Center (GSFC)"
    # add software information
    fileID.software_reference = mdlhmc.version.project_name
    fileID.software_version = mdlhmc.version.full_version


class mask(gravtk.harmonics):
    """
    Inheritance of ``harmonics`` class for masks

    Attributes
    ----------
    lmax: int
        maximum degree of the spherical harmonic field
    mmax: int
        maximum order of the spherical harmonic field
    clm: float
        cosine spherical harmonics
    slm: float
        sine spherical harmonics
    attributes: dict
        attributes of harmonics variables
    shape: tuple
        dimensions of harmonics object
    ndim: int
        number of dimensions of harmonics object
    filename: str
        input or output filename
    """

    np.seterr(invalid="ignore")

    # inherit harmonics class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.band = None

    def to_netCDF4(self, filename, **kwargs):
        """
        Write a harmonics object to netCDF4 file

        Parameters
        ----------
        filename: str
            full path of output netCDF4 file
        title: str or NoneType, default None
            title attribute of dataset
        source: str or NoneType, default None
            source attribute of dataset
        reference: str or NoneType, default None
            reference attribute of dataset
        date: bool, default True
            harmonics objects contain date information
        clobber: bool, default True
            Overwrite an existing netCDF4 file
        verbose: bool, default False
            Output file and variable information
        """
        # set default keyword arguments
        kwargs.setdefault("title", None)
        kwargs.setdefault("source", None)
        kwargs.setdefault("reference", None)
        kwargs.setdefault("date", True)
        kwargs.setdefault("clobber", True)
        kwargs.setdefault("verbose", False)
        # setting NetCDF clobber attribute
        clobber = "w" if kwargs["clobber"] else "a"
        # opening netCDF file for writing
        self.filename = pathlib.Path(filename).expanduser().absolute()
        fileID = netCDF4.Dataset(self.filename, clobber, format="NETCDF4")
        # flatten harmonics
        temp = self.flatten()
        # Defining the netCDF dimensions
        n_harm = len(temp.l)
        n_band = len(np.atleast_1d(temp.band))
        fields = ["l", "m", "clm", "slm", "band"]
        fileID.createDimension("lm", n_harm)
        fileID.createDimension("band", n_band)
        # convert band variable to array
        temp.band = np.atleast_1d(temp.band)
        # defining the netCDF variables
        nc = {}
        # degree and order
        nc["l"] = fileID.createVariable("l", "i", ("lm",))
        nc["m"] = fileID.createVariable("m", "i", ("lm",))
        # spherical harmonics
        nc["clm"] = fileID.createVariable(
            "clm",
            "d",
            (
                "lm",
                "band",
            ),
        )
        nc["slm"] = fileID.createVariable(
            "slm",
            "d",
            (
                "lm",
                "band",
            ),
        )
        # bands
        nc["band"] = fileID.createVariable("band", "i", ("band",))
        # filling netCDF variables
        for key in fields:
            nc[key][:] = getattr(temp, key)
        # Defining attributes for degree and order
        # SH degree long name
        nc["l"].long_name = "spherical_harmonic_degree"
        # SH degree units
        nc["l"].units = "Wavenumber"
        # SH order long name
        nc["m"].long_name = "spherical_harmonic_order"
        # SH order units
        nc["m"].units = "Wavenumber"
        # Defining attributes for harmonics
        nc["clm"].long_name = "cosine_spherical_harmonics"
        nc["slm"].long_name = "sine_spherical_harmonics"
        # Defining attributes for band
        nc["band"].long_name = "band"
        # global variables of NetCDF4 file
        if kwargs["title"]:
            fileID.title = kwargs["title"]
        if kwargs["source"]:
            fileID.source = kwargs["source"]
        if kwargs["reference"]:
            fileID.reference = kwargs["reference"]
        # add software information
        fileID.software_reference = gravtk.version.project_name
        fileID.software_version = gravtk.version.full_version
        # date created
        fileID.date_created = time.strftime("%Y-%m-%d", time.localtime())
        # Output netCDF structure information
        logging.info(self.filename)
        logging.info(list(fileID.variables.keys()))
        # Closing the netCDF file
        fileID.close()

    def flatten(self):
        """
        Flatten harmonics matrices into arrays
        """
        n_harm = (
            self.lmax**2
            + 3 * self.lmax
            - (self.lmax - self.mmax) ** 2
            - (self.lmax - self.mmax)
        ) // 2 + 1
        # restructured degree and order
        temp = mask(lmax=self.lmax, mmax=self.mmax)
        temp.l = np.zeros((n_harm,), dtype=np.int64)
        temp.m = np.zeros((n_harm,), dtype=np.int64)
        # get filenames if applicable
        if getattr(self, "filename"):
            temp.filename = copy.copy(self.filename)
        # copy band variables
        temp.band = np.copy(self.band)
        # restructured spherical harmonic arrays
        if self.clm.ndim == 2:
            temp.clm = np.zeros((n_harm))
            temp.slm = np.zeros((n_harm))
        else:
            n = self.clm.shape[-1]
            temp.clm = np.zeros((n_harm, n))
            temp.slm = np.zeros((n_harm, n))
        # create counter variable lm
        lm = 0
        for m in range(0, self.mmax + 1):  # MMAX+1 to include MMAX
            for l in range(m, self.lmax + 1):  # LMAX+1 to include LMAX
                temp.l[lm] = np.int64(l)
                temp.m[lm] = np.int64(m)
                if self.clm.ndim == 2:
                    temp.clm[lm] = self.clm[l, m]
                    temp.slm[lm] = self.slm[l, m]
                else:
                    temp.clm[lm, :] = self.clm[l, m, :]
                    temp.slm[lm, :] = self.slm[l, m, :]
                # add 1 to lm counter variable
                lm += 1
        # return the flattened arrays
        return temp


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Convert drainage basin masks to
            spherical harmonics and then back to the
            spatial domain after spectral processing
            """
    )
    # command line parameters
    parser.add_argument(
        "infile", type=pathlib.Path, help="Input drainage basin file"
    )
    # ATL15 file for calculating areas
    parser.add_argument(
        "--atl15",
        "-A",
        type=pathlib.Path,
        nargs="+",
        default=[],
        help="ATL15 file to use for areas",
    )
    # maximum spherical harmonic degree and order
    parser.add_argument(
        "--lmax",
        "-l",
        type=int,
        default=60,
        help="Maximum spherical harmonic degree",
    )
    parser.add_argument(
        "--mmax",
        "-m",
        type=int,
        default=None,
        help="Maximum spherical harmonic order",
    )
    # Gaussian smoothing radius (km)
    parser.add_argument(
        "--radius",
        "-R",
        type=float,
        default=0,
        help="Gaussian smoothing radius (km)",
    )
    # Gaussian filter raster image to increase coverage
    parser.add_argument(
        "--sigma",
        "-S",
        type=float,
        default=0.0,
        help="Standard deviation for Gaussian kernel",
    )
    # tolerance in interpolated mask to set as valid
    parser.add_argument(
        "--tolerance",
        "-T",
        type=float,
        default=0.5,
        help="Tolerance to set as valid mask",
    )
    # print information about each input and output file
    parser.add_argument(
        "--verbose",
        "-V",
        action="count",
        default=0,
        help="Verbose output of processing run",
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permission mode of directories and files",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # run program
    convert_basin_masks(
        args.infile,
        args.lmax,
        MMAX=args.mmax,
        RAD=args.radius,
        ATL15=args.atl15,
        SIGMA=args.sigma,
        TOLERANCE=args.tolerance,
        MODE=args.mode,
    )


# run main program
if __name__ == "__main__":
    main()
