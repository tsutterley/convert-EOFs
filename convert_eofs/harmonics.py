"""
harmonics.py
Written by Tyler Sutterley (06/2026)

Inheritance of ``harmonics`` class for EOFs

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html

UPDATE HISTORY:
    Written 06/2026
"""

import copy
import time
import logging
import pathlib
import netCDF4
import numpy as np
import gravity_toolkit as gravtk


class harmonics(gravtk.harmonics):
    """
    Inheritance of ``harmonics`` class for EOFs

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
    num: int or array
        EOF number(s)
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
        self.num = None

    def from_netCDF4(self, filename, **kwargs):
        """
        Read a ``harmonics`` object from a netCDF4 file

        Parameters
        ----------
        filename: str
            full path of input netCDF4 file
        date: bool, default True
            netCDF4 file has date information
        compression: str or NoneType, default None
            file compression type

                - ``'gzip'``
                - ``'zip'``
                - ``'bytes'``
        verbose: bool, default False
            print file and variable information
        """
        # set filename
        self.case_insensitive_filename(filename)
        # Open the NetCDF4 file for reading
        fileID = netCDF4.Dataset(self.filename, mode="r")
        # Output NetCDF file information
        logging.info(fileID.filepath())
        logging.info(list(fileID.variables.keys()))
        # read flattened spherical harmonics
        temp = harmonics()
        temp.filename = copy.copy(self.filename)
        # create list of variables to retrieve
        fields = ["l", "m", "clm", "slm", "num"]
        # Getting the data from each NetCDF variable
        for field in fields:
            setattr(temp, field, fileID.variables[field][:].copy())
        # calculate maximum degree and order
        temp.lmax = np.max(temp.l)
        temp.mmax = np.max(temp.m)
        # expand the spherical harmonics to dimensions
        self = temp.expand()
        self.num = temp.num.copy()
        # attributes of clm/slm and included variables
        for key in fields:
            # attempt to get attribute for variable
            try:
                self.attributes[key] = [
                    fileID.variables[key].units,
                    fileID.variables[key].long_name,
                ]
            except (KeyError, ValueError, AttributeError):
                pass
        # get global netCDF4 attributes
        self.attributes["ROOT"] = {}
        for att_name in fileID.ncattrs():
            self.attributes["ROOT"][att_name] = fileID.getncattr(att_name)
        # Closing the NetCDF file
        fileID.close()
        return self

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
        n_EOF = len(np.atleast_1d(temp.num))
        fields = ["l", "m", "clm", "slm", "num"]
        fileID.createDimension("lm", n_harm)
        fileID.createDimension("num", n_EOF)
        # convert EOF number variable to array
        temp.num = np.atleast_1d(temp.num)
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
                "num",
            ),
        )
        nc["slm"] = fileID.createVariable(
            "slm",
            "d",
            (
                "lm",
                "num",
            ),
        )
        # EOF number
        nc["num"] = fileID.createVariable("num", "i", ("num",))
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
        # Defining attributes for EOF number
        nc["num"].long_name = "EOF number"
        nc["num"].units = "unitless"
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
        temp = harmonics(lmax=self.lmax, mmax=self.mmax)
        temp.l = np.zeros((n_harm,), dtype=np.int64)
        temp.m = np.zeros((n_harm,), dtype=np.int64)
        # get filenames if applicable
        if getattr(self, "filename"):
            temp.filename = copy.copy(self.filename)
        # copy EOF variables
        temp.num = np.copy(self.num)
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
