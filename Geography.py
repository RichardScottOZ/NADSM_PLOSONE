import numpy as np
from Constants import *

# Simple tests for broad regions
def Europe(lat,lon):
    # stops just before moscow
    return (-17 < lon <  35 and lat >  35 ) and not Anatolia(lat,lon)

def Russia(lat,lon):
    # Central and Western Russia
    return (35 <= lon <= 89 and lat >= 35) and not Anatolia(lat,lon) and not Mesopotamia(lat,lon)

def Britain(lat,lon):
    return (lon <= 1 and lat >= 50 )

def Anatolia(lat,lon):
    # anatolia and the caucus
    return (25 <= lon <= 48 and 35  <= lat <= 41 )

def Egypt(lat,lon):
    return (28 <= lon <= 35 and 16  <= lat <= 31 )

def Mesopotamia(lat,lon):
    return (30 < lon <  61 and 16  < lat <= 41) or Anatolia(lat,lon) or Egypt(lat,lon)

def North_Africa(lat,lon):
    return (-14 <= lon <= 28 and 26  <= lat <= 36 )

def sub_Saharan_Africa(lat,lon):
    # -4 lat to match/overlap the start of Agr_Expand1 in Africa
    return (-17 <= lon <= 53 and -3  <  lat <= 16 )

def Africa(lat,lon):
    return North_Africa(lat,lon) or sub_Saharan_Africa(lat,lon) or Egypt(lat,lon)

def India(lat,lon):
    return (61 <= lon <= 89 and 0  <= lat <  35 )

def Asia(lat,lon):
    # China but drop tundra to the north; include Khmer and Indonesia, etc. to the south
    return (lon >  89 and -13  < lat <= 47)

def China(lat,lon):
    # drops Khmer etc to the south; drop tundra to the north
    return (lon >  89 and 17  < lat <= 47)

def LiaoRV(lat,lon):
    return (95 <= lon <= 180 and  40  <  lat <= 70)
# def MYRV(lat,lon):
#     return (95 <= lon <= 180 and  30  <  lat <= 40) 
# def PearlRV(lat,lon):
#     return (95 <= lon <= 180 and  19  <  lat <= 30)
def MekongRV(lat,lon):
    return (95 <= lon <= 180 and -15  <  lat <= 19)
def BurmaRV(lat,lon):
    return (85 <= lon <=  95 and  20  <  lat <= 40)

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
myrv_shape = None
def MYRV(lat,lon):
    global myrv_shape
    if myrv_shape is None:
        # note diagnonal downward base (could be any shape w/ many vertices)
        myrv_shape = Polygon([(95,30),(150,25),(150,40),(95,40)])
    return myrv_shape.contains(Point(lon,lat))

pearlrv_shape = None
def PearlRV(lat,lon):
    global pearlrv_shape
    if pearlrv_shape is None:
        # note diagnonal upward base to abutt MYRV above
        pearlrv_shape = Polygon([(95,19),(150,19),(150,25),(95,30)])
    return pearlrv_shape.contains(Point(lon,lat))


geography_region_list = ['Europe','Anatolia','Egypt','Mesopotamia','North_Africa','sub_Saharan_Africa','Africa','Asia','China','Russia','Britain','India',
                         'LiaoRV','MYRV','PearlRV','MekongRV','BurmaRV']

class Geography:
    def __init__(self,name,n_regions):
        self.name = name
        self.n_regions = n_regions
        self.lat = np.zeros(n_regions)
        self.lon = np.zeros(n_regions)
        self.Elevation = np.zeros(n_regions)
        self.Biome = np.zeros(n_regions)
        self.Biome[:] = GEO_DESERT

        # region sizes and distances (in km)
        
        # Littoral regions boolean (needed?)
        # immediate von Neuman connections to *all regions* (regardless of biome)
        # when forming territories a subset might be chosen, such as only AG possible, etc.
        # LandConnection {region => [regions]}, including negative numbers to indicate ocean at some VN neighbor
        # SeaConnection {littoral_region => {[[lit_region dist], ...]}) sorted by increasing distance (encode as km)

        # The actual empire history of these regions, if known empire_index[year,region]
        #DEAD self.history = None # up to the creator to provide this, if appropriate
        self.actual_history = False
        self.min_polity_size = 0
        self.years = None
        self.empire_names = None # names per empire index with [0] ensured as '' (e.g., Hinterland, Unoccupied)

    def __repr__(self):
        return "<Geography %s (%d regions)>" % (self.name,self.n_regions)
        
