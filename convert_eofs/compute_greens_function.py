#!/usr/bin/env python
"""
compute_greens_function.py
Written by Tyler Sutterley (10/2026)
Computes Green's function kernels

COMMAND LINE OPTIONS:
    --help: list the command line options
    -l X, --lmax X: maximum spherical harmonic degree
    -m X, --earth-model X: Earth model
    --spacing X Y: grid spacing in x and y directions (meters)
    --width X Y: grid width in x and y directions (meters)
    -O X, --output-directory X: output directory
    -V, --verbose: Verbose output of processing run
    --log: Output log of files created for each job
    -M X, --mode X: Permissions mode of the files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html

REFERENCES:
    W. E. Farrell, "Deformation of the Earth by Surface Loads",
        Reviews of Geophysics, 10(3), 761-797, 1972.
        https://doi.org/10.1029/RG010i003p00761
    W. E. Farrell, "Earth Tides, Ocean Tides and Tidal Loading",
        Philosophical Transactions for the Royal Society of London,
        274(1239), 253-259, 1973. https://doi.org/10.1098/rsta.1973.0050

UPDATE HISTORY:
    Updated 10/2026: use default file logger for valid and failed program runs
        use new structured netCDF4 output function from geoid-toolkit
    Written 11/2024
"""

import os
import sys
import pathlib
import argparse
import logging
import traceback
import numpy as np
import gravity_toolkit as gravtk
import model_harmonics as mdlhmc
import geoid_toolkit as geoidtk


# PURPOSE: keep track of threads
def info(args):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info(pathlib.Path(sys.argv[0]).name)
    logger.info(args)
    logger.info(f"module name: {__name__}")
    if hasattr(os, "getppid"):
        logger.info(f"parent process: {os.getppid():d}")
    logger.info(f"process id: {os.getpid():d}")


def compute_greens_function(
    LMAX,
    earth_model,
    spacing=[],
    width=[],
    output_directory=None,
    mode=0o775,
):
    # create output directory if it does not exist
    output_directory = pathlib.Path(output_directory).expanduser().absolute()
    output_directory.mkdir(mode=mode, parents=True, exist_ok=True)

    # parameters for load Love numbers
    citation = "Wang et al. (2012)"
    reference = "CF"
    # use "complete" load Love numbers to high-degree and order
    love_numbers_file = gravtk.utilities.get_cache_path(
        f"{earth_model}-LLNs-complete.dat"
    )
    header = 1
    columns = ["l", "hl", "ll", "kl", "nl", "nk"]
    # read load Love numbers
    hl, kl, ll = gravtk.read_love_numbers(
        love_numbers_file,
        HEADER=header,
        COLUMNS=columns,
        LMAX=LMAX,
        REFERENCE=reference,
        FORMAT="tuple",
    )

    # dictionary describing the output netCDF4 structure
    struct = dict(dimensions=("y", "x"), variables={})
    struct["variables"]["G"] = ("y", "x")

    # global attributes
    attributes = dict(ROOT={})
    attributes["ROOT"]["institution"] = (
        "NASA Goddard Space Flight Center (GSFC)"
    )
    attributes["ROOT"]["project"] = "GSFC-fdm"
    attributes["ROOT"]["product_type"] = "gravity_field"
    attributes["ROOT"]["title"] = "Green's function kernel"
    # add attributes for LMAX
    attributes["ROOT"]["max_degree"] = LMAX
    # add attributes for earth model and love numbers
    attributes["ROOT"]["earth_model"] = earth_model
    attributes["ROOT"]["earth_love_numbers"] = citation
    attributes["ROOT"]["reference_frame"] = reference
    # add attributes for earth parameters
    factors = gravtk.units()
    attributes["ROOT"]["earth_radius"] = f"{factors.rad_e / 1e2:0.3f} m"
    attributes["ROOT"]["earth_density"] = f"{factors.rho_e * 1e3:0.3f} kg/m^3"
    attributes["ROOT"]["earth_gravity_constant"] = (
        f"{factors.GM / 1e6:0.3f} m^3/s^2"
    )
    # Defining attributes for x and y coordinates
    attributes["x"] = dict(
        long_name="Easting",
        standard_name="kernel_x_coordinate",
        description="X-coordinates of the kernel",
        units="meters",
    )
    attributes["y"] = dict(
        long_name="Northing",
        standard_name="kernel_y_coordinate",
        description="Y-coordinates of the kernel",
        units="meters",
    )
    # Defining attributes for the Green's function kernel variable
    attributes["G"] = dict(
        long_name="kernel",
        description="Green's function kernel",
        units="m/kg",
    )

    # grid spacing
    dx, dy = np.broadcast_to(np.atleast_1d(spacing), (2,))
    # grid width
    W = np.broadcast_to(np.atleast_1d(width), (2,))

    # generate Green's function kernel
    Xk, Yk, G = mdlhmc.greens_kernel(
        LMAX, SPACING=(dx, dy), WIDTH=W, LOVE=(hl, kl, ll)
    )
    # output dictionary
    output = dict(G=G, x=Xk, y=Yk)

    # output to netCDF4 file
    kernel_file = f"Gkernel_{earth_model}_L{LMAX:d}_{dx / 1e3:02.0f}km.nc"
    output_file = pathlib.Path(output_directory).joinpath(kernel_file)
    # write data to netCDF4 file
    geoidtk.spatial.to_netCDF4(
        output,
        attributes,
        output_file,
        data_type="structured",
        structure=struct,
    )
    # set permissions mode of the output file
    output_file.chmod(mode)
    # return the output filename
    return output_file


def arguments():
    # create an argument parser
    parser = argparse.ArgumentParser(
        description="Computes Green's function kernels"
    )
    # command line parameters
    # spherical harmonic degree
    parser.add_argument(
        "--lmax",
        "-l",
        type=int,
        help="Maximum spherical harmonic degree",
    )
    # earth model
    models = [
        "ak135",
        "ak135hard",
        "iasp91",
        "iasp91hard",
        "PREM",
        "PREMhard",
        "PREMsoft",
    ]
    parser.add_argument(
        "--earth-model",
        "-m",
        type=str,
        choices=models,
        default="PREM",
        help="Earth model",
    )
    # grid spacing and width
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=2,
        default=(1e3, 1e3),
        help="Grid spacing (meters)",
    )
    parser.add_argument(
        "--width",
        type=float,
        nargs=2,
        default=(160e3, 160e3),
        help="Grid width (meters)",
    )
    # working data directory
    parser.add_argument(
        "--output-directory",
        "-O",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Working data directory",
    )
    # verbose output
    parser.add_argument(
        "--verbose",
        "-V",
        action="count",
        default=0,
        help="Verbose output of processing run",
    )
    # Output log file for each job in forms
    # validrun_2002-04-01T00:00:00_PID-00000.log
    # failedrun_2002-04-01T00:00:00_PID-00000.log
    parser.add_argument(
        "--log",
        default=False,
        action="store_true",
        help="Output log file for each job",
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="permissions mode of output files",
    )
    # return the parser
    return parser


# main function
def main():
    # parse the command line arguments
    parser = arguments()
    args = parser.parse_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logger = gravtk.utilities.build_logger(
        __name__, level=loglevels[args.verbose]
    )

    # try to run the program with listed parameters
    try:
        info(args)
        # compute Green's function kernel
        output_file = compute_greens_function(
            args.lmax,
            args.earth_model,
            spacing=args.spacing,
            width=args.width,
            output_directory=args.output_directory,
            mode=args.mode,
        )
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logger.critical(f"process id {os.getpid():d} failed")
        logger.error(traceback.format_exc())
        if args.log:  # write failed job completion log file
            logfile = gravtk.utilities.create_log_file(
                "failedrun",
                filename=pathlib.Path(sys.argv[0]).name,
                arguments=vars(args),
            )
            logger.info(logfile)
    else:
        if args.log:  # write successful job completion log file
            logfile = gravtk.utilities.create_log_file(
                "validrun",
                filename=pathlib.Path(sys.argv[0]).name,
                arguments=vars(args),
                output=output_file,
            )
            logger.info(logfile)


# run main program
if __name__ == "__main__":
    main()
