#!/usr/bin/env python
"""
convert_atl11_atl15_heights.py
Written by Tyler Sutterley (03/2023)
Read monthly height variables and convert to spherical harmonics
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
import gravity_toolkit as gravtk
import model_harmonics as mdlhmc

# ignore pyproj and divide by zero warnings
warnings.filterwarnings("ignore")


# PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if REGION in ("ais",):
        projection_flag = "EPSG:3031"
    elif REGION in ("gris",):
        projection_flag = "EPSG:3413"
    return projection_flag


# PURPOSE: read a variable group from ICESat-2 ATL15
def read_ATL15(infile, group="delta_h", fields=None):
    # dictionary with ATL15 variables
    ATL15 = {}
    attributes = {}
    infile = pathlib.Path(infile).expanduser().absolute()
    with netCDF4.Dataset(infile, mode="r") as fileID:
        # check if reading from root group or sub-group
        ncf = fileID.groups[group] if group else fileID
        for key in fields or ncf.variables.keys():
            val = ncf.variables[key]
            ATL15[key] = val[:]
            attributes[key] = {}
            for att_name in val.ncattrs():
                attributes[key][att_name] = val.getncattr(att_name)
    # return the data and attributes
    return (ATL15, attributes)


# PURPOSE: create a mosaic from ICESat-2 ATL15 files
def mosaic_ATL15(filenames, group="delta_h", fields=["ice_area"]):
    # parse ATL15 file
    pattern = r"(ATL\d{2})_(.*?)_(\d{2})(\d{2})_(.*?)_(\d{3})_(\d{2}).nc$"
    rx = re.compile(pattern, re.VERBOSE)
    # create mosaic of ATL15 data
    mosaic = mdlhmc.spatial.mosaic()
    # iterate over each ATL15 file
    for f in filenames:
        # verify path to input ATL15 file
        f = pathlib.Path(f).expanduser().absolute()
        PRD, RGN, SCYC, ECYC, RES, RL, VERS = rx.findall(f.name).pop()
        DIRECTORY = pathlib.Path(f.parent).expanduser().absolute()
        # get ATL15 dimension variables from group
        d, attrib = read_ATL15(f, group=group, fields=["x", "y", "time"])
        # update the mosaic grid spacing
        mosaic.update_spacing(d["x"], d["y"])
        mosaic.update_bounds(d["x"], d["y"])
    # dimensions of output mosaic
    ny, nx = mosaic.shape
    nt = len(d["time"])
    # create output mosaic
    output = {}
    output["x"] = np.copy(mosaic.x)
    output["y"] = np.copy(mosaic.y)
    output["time"] = np.copy(d["time"])
    valid_mask = np.zeros((nt, ny, nx), dtype=bool)
    for key in fields:
        output[key] = np.ma.zeros((nt, ny, nx))
    # iterate over each ATL15 file
    for f in filenames:
        # get ATL15 variables from group
        d, attrib = read_ATL15(f, group=group)
        # netCDF4 structure information
        logging.debug(str(f))
        logging.debug(list(d.keys()))
        # get the image coordinates of the input file
        iy, ix = mosaic.image_coordinates(d["x"], d["y"])
        valid_mask[:, iy, ix] |= True
        for key in fields:
            output[key].fill_value = attrib[key]["_FillValue"]
            output[key][:, iy, ix] = d[key][...]
    # update masks for variables
    for key in fields:
        val = output[key]
        val.mask = (
            (val.data == val.fill_value)
            | np.isnan(val.data)
            | np.logical_not(valid_mask)
        )
        val.data[val.mask] = val.fill_value
    # return the output mosaic as a dictionary
    return output


def convert_atl11_atl15_heights(
    INPUT_FILE,
    LMAX,
    MMAX=None,
    RAD=0,
    MASKS=None,
    ATL15=None,
    BUFFER=800e3,
    MODE=0o775,
):
    # verify input file
    INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    if not INPUT_FILE.exists():
        raise FileNotFoundError("IS2 file not found in file system")
    # read height file and extract variables
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
    fv = np.float64(fileID.variables["height"]._FillValue)
    # input variable units
    variable_units = fileID.variables["height"].units
    # variable fields (check if additional error fields)
    fields = sorted(
        set(fileID.variables.keys()) - set(fileID.dimensions.keys())
    )
    logging.debug(f"fields:{','.join(fields)}")
    # fix Greenland ATL15 coordinates
    dx = np.abs(fd["x"][1] - fd["x"][0])
    if (REGION.lower() == "gris") and (dx != 10e3):
        fd["x"] = fd["x"][0] + 10e3 * np.arange(len(fd["x"]))
    # calculate grid areas (assume fully ice covered)
    dx = np.abs(fd["x"][1] - fd["x"][0])
    dy = np.abs(fd["y"][1] - fd["y"][0])
    # input shape of input data
    nt, ny, nx = np.shape(fd["height"])
    shape = (ny, nx)
    output_shape = (ny + int(2 * BUFFER // dy), nx + int(2 * BUFFER // dx))
    indexing = "xy"
    logging.debug(f"Shape: {shape}")
    logging.debug(f"Output shape: {output_shape}")
    # extract x and y coordinate arrays
    xg, yg = np.meshgrid(fd["x"], fd["y"], indexing=indexing)
    # close the netCDF4 file
    fileID.close()

    # create mask object for reducing data
    if not MASKS:
        fd["mask"] = np.ones((shape), dtype=bool)
    else:
        fd["mask"] = np.zeros((shape), dtype=bool)
    # read masks for reducing regions before converting to harmonics
    for mask_file in MASKS:
        logging.info(mask_file)
        fileID = netCDF4.Dataset(mask_file, mode="r")
        fd["mask"] |= fileID.variables["mask"][:].astype(bool)
        fileID.close()
    # indices of valid ATL11/ATL15 data
    fd["mask"] &= np.any((fd["height"] != fv), axis=0)
    fd["mask"] &= np.any(np.isfinite(fd["height"]), axis=0)

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
    output = dict(time=np.copy(fd["time"]))
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

    # reduce latitude and longitude to valid and masked points
    indx, indy = np.nonzero(fd["mask"])
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
    for var in fields:
        # allocate for output spherical harmonics
        Ylms = gravtk.harmonics(lmax=LMAX, mmax=MMAX)
        Ylms.clm = np.zeros((LMAX + 1, MMAX + 1, nt))
        Ylms.slm = np.zeros((LMAX + 1, MMAX + 1, nt))
        Ylms.time = np.copy(fd["time"])
        Ylms.month = gravtk.time.calendar_to_grace(fd["time"])

        # # input spherical harmonic data file for variable
        # prefix = re.sub(r'Height', var.title(), PREFIX, re.IGNORECASE)
        # FILE = f'{prefix}_{REGION}_CLM_L{LMAX:d}{order_str}.nc'
        # CLM_FILE = INPUT_FILE.with_name(FILE)
        # Ylms = gravtk.harmonics().from_netCDF4(CLM_FILE)

        # output spatial
        output[var] = np.ma.zeros((nt, *output_shape), fill_value=fv)
        # for each time step
        for n in range(nt):
            # reduce data to time and scale areas
            # note that some values at time points will be 0
            # due to the time-variable masking of the ATL15 data
            SCALED = np.nan_to_num(scaling_factors * fd[var][n, indx, indy], 0)
            # convert scaled values to spherical harmonics
            # use custom UNITS to keep as inputs but use 4-pi norm
            YLMS = gravtk.gen_point_load(
                SCALED, lon, lat, LMAX=LMAX, MMAX=MMAX, UNITS=UNITS
            )
            # copy spherical harmonics for time
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
            # reshape to output and save for time
            if var in ("rand_e", "sys_e"):
                output[var][n, :, :] = np.abs(spatial.reshape(output_shape))
            else:
                output[var][n, :, :] = spatial.reshape(output_shape)

        # output spherical harmonic data file for variable
        prefix = re.sub(r"Height", var.title(), PREFIX, re.IGNORECASE)
        FILE = f"{prefix}_{SEP}{REGION}_CLM_L{LMAX:d}{order_str}.nc"
        CLM_FILE = INPUT_FILE.with_name(FILE)
        Ylms.to_netCDF4(CLM_FILE, **attributes)
        # change the permissions mode of the output file to MODE
        CLM_FILE.chmod(mode=MODE)

    # output ATL11/ATL15 data file
    FILE = f"{PREFIX}_{SEP}{REGION}{AUX}_L{LMAX:d}{order_str}{gw_str}.nc"
    OUTPUT_FILE = INPUT_FILE.with_name(FILE)
    output_to_netCDF4(OUTPUT_FILE, output, region=REGION, fields=fields)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)


# PURPOSE: output gridded data to netCDF4
def output_to_netCDF4(output_file, output, **kwargs):
    # set default keyword arguments
    kwargs.setdefault("region", "gris")
    kwargs.setdefault("fields", ["height"])

    # opening NetCDF file for writing
    logging.info(str(output_file))
    fileID = netCDF4.Dataset(output_file, "w", format="NETCDF4")

    # output shape of data
    nt, ny, nx = np.shape(output["height"])
    dims = (
        "time",
        "y",
        "x",
    )

    # Defining the NetCDF dimensions
    fileID.createDimension("x", nx)
    fileID.createDimension("y", ny)
    fileID.createDimension("time", nt)

    # python dictionary with netCDF4 variables
    nc = {}
    # defining the NetCDF variables
    nc["x"] = fileID.createVariable("x", output["x"].dtype, ("x",))
    nc["y"] = fileID.createVariable("y", output["y"].dtype, ("y",))
    nc["time"] = fileID.createVariable("time", output["time"].dtype, ("time",))
    # for each output variable
    for v in kwargs["fields"]:
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
    for v in kwargs["fields"]:
        # set variable attributes
        nc[v].setncattr("units", "meters")
        nc[v].setncattr("standard_name", v)
        # set grid mapping attribute
        nc[v].setncattr("grid_mapping", "Polar_Stereographic")

    # global attributes of output netCDF4 file
    fileID.title = "ICESat-2 ATL11 ATL15 variables"
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


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Convert ATL11/ATL15 heights to
            spherical harmonics and then back to the
            spatial domain after spectral processing
            """
    )
    # command line parameters
    parser.add_argument(
        "infile", type=pathlib.Path, help="Input ATL11/ATL15 data file"
    )
    # mask file for reducing to regions
    parser.add_argument(
        "--mask",
        type=pathlib.Path,
        nargs="+",
        default=[],
        help="netCDF4 masks file for reducing to regions",
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
    convert_atl11_atl15_heights(
        args.infile,
        args.lmax,
        MMAX=args.mmax,
        RAD=args.radius,
        MASKS=args.mask,
        ATL15=args.atl15,
        MODE=args.mode,
    )


# run main program
if __name__ == "__main__":
    main()
