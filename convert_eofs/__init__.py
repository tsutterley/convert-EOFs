from .ATL15 import read_ATL15, mosaic_ATL15
from .harmonics import eof, mask


def crs_to_cf(region):
    """Dictionary defining the spatial reference system for each region"""
    # dictionary defining the spatial reference system for each region
    CRS = dict(gris={}, ais={})
    # Greenland (EPSG:3413)
    CRS["gris"]["standard_name"] = "Polar_Stereographic"
    CRS["gris"]["grid_mapping_name"] = "polar_stereographic"
    CRS["gris"]["straight_vertical_longitude_from_pole"] = -45.0
    CRS["gris"]["latitude_of_projection_origin"] = 90.0
    CRS["gris"]["standard_parallel"] = 70.0
    CRS["gris"]["scale_factor_at_projection_origin"] = 1.0
    CRS["gris"]["false_easting"] = 0.0
    CRS["gris"]["false_northing"] = 0.0
    CRS["gris"]["semi_major_axis"] = 6378.137
    CRS["gris"]["semi_minor_axis"] = 6356.752
    CRS["gris"]["inverse_flattening"] = 298.257223563
    CRS["gris"]["spatial_epsg"] = "3413"
    # Antarctica (EPSG:3031)
    CRS["ais"]["standard_name"] = "Polar_Stereographic"
    CRS["ais"]["grid_mapping_name"] = "polar_stereographic"
    CRS["ais"]["straight_vertical_longitude_from_pole"] = 0.0
    CRS["ais"]["latitude_of_projection_origin"] = -90.0
    CRS["ais"]["standard_parallel"] = -71.0
    CRS["ais"]["scale_factor_at_projection_origin"] = 1.0
    CRS["ais"]["false_easting"] = 0.0
    CRS["ais"]["false_northing"] = 0.0
    CRS["ais"]["semi_major_axis"] = 6378.137
    CRS["ais"]["semi_minor_axis"] = 6356.752
    CRS["ais"]["inverse_flattening"] = 298.257223563
    CRS["ais"]["spatial_epsg"] = "3031"
    return CRS[region]
