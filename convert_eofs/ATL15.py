"""
ATL15.py
Written by Tyler Sutterley (06/2023)
Read and mosaic ICESat-2 ATL15 files

UPDATE HISTORY:
    Written 06/2026: merged from individual scripts into singular
"""
import re
import logging
import netCDF4
import pathlib
import numpy as np
import model_harmonics as mdlhmc


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
