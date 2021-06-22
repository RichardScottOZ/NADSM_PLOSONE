# A collection of constants and controls used over the entire set of models

# If we use sets for some critical data structures it is faster BUT
# their order depends on when things are run (and not the random number generator explicitly)

# This is because the elements in sets must be hashable and the std hash uses
# the memory address and this defines the order in the set.  Running in the same
# python instance yields different addresses for objects, hence different order
# of execution.  But if you use lists, the order is well-defined even if some
# tests are slower.

# Thus for debugging set use_set_versions = False; slower but replicable
# use_set_versions = True # CONTROL FASTER but not replicable
use_set_versions = False # CONTROL these are O(n^2) versions

# NOTE You must set use_set_versions before importing Utils!!
from Utils import * # Oxford_comma 



# These correspond to the tags in the data from Turchin and Currie
# Coarse geographical biome types of regions
GEO_DESERT  = 0 # where camels roam
GEO_STEPPES = 1 # where horses roam
GEO_RIVER   = 2 # where fish roam
GEO_AGRICULTURAL = 3 # where farmers roam
# GEO_DESERT near GEO_RIVER become GEO_AGRICULTURAL
# Use these to control which Biomes to compare, etc.
all_biomes  = [GEO_DESERT, GEO_STEPPES, GEO_RIVER, GEO_AGRICULTURAL]
agri_biomes = [GEO_AGRICULTURAL]
agst_biomes = [GEO_AGRICULTURAL, GEO_STEPPES]
astd_biomes = [GEO_AGRICULTURAL, GEO_STEPPES, GEO_DESERT]
def biome_types(biome_type):
    # lists are hashable, sigh
    if biome_type is all_biomes:
        return 'all'
    else:
        types = []
        for biome in biome_type:
            types.append({GEO_DESERT:'desert',GEO_AGRICULTURAL:'agricultural',GEO_STEPPES:'steppe',GEO_RIVER:'riverine'}[biome])
        return Oxford_comma(types)

century = 100
million = 1e6

# States of polities and confederations and their members
POLITY_ALIVE     = 1
POLITY_MEMBER    = 2 # member of a confederation
# These are all different types of polity deaths...
POLITY_DEAD      = 3 # no territories
POLITY_ANNEXED   = 4 # like POLITY_DEAD but had some territories at some point but doesn't any more
POLITY_COLLAPSED = 5 # explicitly collapsed

polity_state_tags = {POLITY_ALIVE:     'A',
                     POLITY_MEMBER:    'M',
                     POLITY_DEAD:      'D',
                     POLITY_ANNEXED:   'X',
                     POLITY_COLLAPSED: 'C',
                     }
# The PNAS regions are not equal-sided, with nearly 2x size longitudinaly:
# analysis of Tom's lat/lon assignments
# this drives a 20Kkm^2 region on the equator
PNAS_LONGITUDE_SCALE = 1.7966 # degrees/PNAS region 'longitude'
PNAS_LATITUDE_SCALE  = 0.8983 # degrees/PNAS region 'latitude'
# diagonal distance across a region in degrees
# math.sqrt(PNAS_LONGITUDE_SCALE*PNAS_LONGITUDE_SCALE + PNAS_LATITUDE_SCALE*PNAS_LATITUDE_SCALE) = 2.0087
# So rather than 1Mha per region there are really
# PNAS_LONGITUDE_SCALE*PNAS_LATITUDE_SCALE = 1.6139 Mha per region

# Polity types for NADSM work
POLITY_TYPE_AG = 1 # Agrarian
POLITY_TYPE_NT = 2 # Nomadic tribe
POLITY_TYPE_NC = 3 # Nomadic confederation
