#! /usr/bin/env python

# Nomadic-Agrarian Demographic-Structural Model

from WHM import *
whm = None # package global
geo = None # package global
from PopElites import * 
from ComparativeHx import *
from Geography import *
from Utils import *
import numpy as np
import copy
import pprint

from PopChronicler import *
from Constants import *
# from decorators import debug_on

# Dump detailed polity state_of_the_union() stats (PE:) and annexation stats (X:)
dump_stats = False

# Report annexations in trial data
report_polity_history = True # need this for a lot of scripts
report_annexations = False # if True creates large .mat files around 9Mb
print_collapse_history = False

# because of 'seams' of agricultural intensification changes, especially in time,
# it is possible for one hinterland region to suddenly become more powerful wrt another hinterland region
# this is an artifactual bug based on crude, not smoothed intensification changes
avoid_hinterland_hinterland_annexation = False # normally False

# Under VN assumptions, each hinterland polity would have 4 connections
# In reality, we need to compute this value when hinterland is first formed
# and before any other states are asserted.  We re-compute that below in setup_for_trial()
# MATLAB: (length(find(geo.land_connections > 0))/prod(size(geo.land_connections)))*size(geo.land_connections,2)
# For OldWorld = 3.7506
hinterland_border_cost = 4 # default worst case value

generation_years = 25 # Number of years in a generation

# Stats from Turchin data to help scale size of nomadic tribes and confederations
# 418 steppe regions; 951 tribal desert regions within 500km of steppe
OldWorld_number_of_steppe_regions = 413 # CONSTANT from OldWorld.pkl
OldWorld_number_of_tribal_regions = 935 # CONSTANT includes GEO_STEPPES and GEO_DESERT within tribal_distance_from_steppe=500
# I estimate there where at most 30 tribes in the entire Eurasian steppe region at any one time
OldWorld_number_of_tribes = 30
# However we construct them randomly and can't always get 30 tribes...a few more smaller tribes at periphery exist
# We assume there will be about 30*1.5 = 45 tribes created on average in the OldWorld
OldWorld_number_of_tribes_created = OldWorld_number_of_tribes*1.5
# Longitudes of various geographic locations in the steppe world of interest
Ural_lon  = 54 # degrees E
Altai_lon = 90 # degrees E
Pontic_Caspian_lon = 45.0 # between 40 and 50 really
China_steppe_border_lon = 112.0
horse_cavalry_diffusion_distance = (China_steppe_border_lon - Pontic_Caspian_lon)*111.32*np.cos(np.radians(40))

# tags for composing tribes and factions
AVAILABLE = -1
UNAVAILABLE = 0
ASSIGNED = 1 # assigned is an id > 0

class War(Order): # Order is Reusable; see WHM.py
    def __init__(self,us_territory,them_territory):
        self.us_territory = us_territory
        self.us_polity = us_territory.polity
        self.them_territory = them_territory
        self.them_polity = them_territory.polity
        if us_territory is them_territory:
            raise RuntimeError("How can we be attacking ourselves?")

    def __repr__(self):
        # return "War(%s vs. %s)" % (self.us_polity,self.them_polity)
        return "War (%s@%s vs. %s@%s)" % (self.us_polity,self.us_territory,
                                          self.them_polity,self.them_territory)
    
    def execute(self):
        us   = self.us_territory.polity
        them = self.them_territory.polity
        # Check that this order is still valid
        if us is not self.us_polity or them is not self.them_polity:
            return # rescind this order
        
        global whm,geo
        global hinterland_border_cost
        Hinterland = whm.Hinterland
        battle_i = self.them_territory.t_index
        # We separate the 'raw_power' from the elite population in case later we use
        # a more complicated asabiya story to pump up or depress raw_power, separate from actual people
        # Compute the power of us projected to the battle territory (offence)
        if us is Hinterland:
            # Single region territory projected 1 region away
            # Each regions's capital is itself
            pp_us = whm.power_scale_1[self.them_territory.lat]
            # hinterland's raw power is distributed over its (4) political borders
            raw_power_us = whm.raw_territorial_power[self.us_territory.t_index]/hinterland_border_cost
            us_i = [self.us_territory.t_index]
        else:
            pp_us = us.power_scale[battle_i]
            # already scaled by number of political borders in update_power()
            raw_power_us = us.raw_power
            us_i = us.territories_i
        power_us = raw_power_us *pp_us # ADSM eqn 6

        # Compute the power of them projected to the battle territory (defense)
        if them is Hinterland:
            # Single region territory defense of self -- no distance factor
            pp_them = 1
            # See comment about hinterland borders above
            raw_power_them = whm.raw_territorial_power[battle_i]/hinterland_border_cost
            them_i = [battle_i]
        else:
            pp_them = them.power_scale[battle_i]
            # already scaled by number of political borders in update_power()
            raw_power_them = them.raw_power
            them_i = them.territories_i
        power_them = raw_power_them*pp_them # ADSM eqn 6

        # Power politics
        total_elites_us   = np.sum(whm.elites[us_i])
        total_elites_them = np.sum(whm.elites[them_i])
        army_us   = total_elites_us  *pp_us
        army_them = total_elites_them*pp_them

        if power_us == 0 or army_us == 0:
            power_ratio = 0 # we can't win; if they have zero no one wins
        elif power_them == 0 or army_them == 0:
            power_ratio = whm.War_victory # us must have power but them don't so us wins
        else:
            power_ratio = power_us/power_them

        if power_ratio >= whm.War_victory: # ADSM eqn 8
            # annexation!

            if False and whm.display_show_figures:
                # CHRONICLER! Show which regions of the territory were exchanged
                # This display is temporary until the figure is redrawn each display_time_interval
                # This works ONLY because orders are executed after the state of the union
                # chronicle call runs and shows where the polities are
                regions = self.them_territory.regions
                whm.display_ax_p.plot(geo.lon[regions],geo.lat[regions], marker='o',linestyle='None',color='Red')
                whm.display_fig_p.canvas.draw() # force out drawing

            # ADSM pop TODO: us loses a few elites but them loses a large fraction their elites assigned to this war
            # this should help make migration of us into vanquished_territory easier -- opportunities abound
            # the dead are uniformly (not proportionately?) culled from all the territories BEFORE annexation
            vanquished_dead = army_them*whm.vanquished_death_fraction # what fraction of their army dies?
            # what fraction of our army dies?  A fraction of their deaths
            victorious_dead = vanquished_dead*whm.victorious_death_fraction # will always be less than vanquished_dead

            # death of elites proportional to regional contribution
            if vanquished_dead and total_elites_them: # in case there aren't any elites (see above)
                whm.elites[them_i] -= vanquished_dead*(whm.elites[them_i]/total_elites_them)
            if victorious_dead and total_elites_us:
                whm.elites[us_i]   -= victorious_dead*(whm.elites[us_i]  /total_elites_us)
                
            # if us is Hinterland then this will cause a new state to arise, hence us_alt
            us_alt = us.annex(self.us_territory,self.them_territory,update_state=False,hinterland_arise=whm.hinterland_arise)
            cw = us.previous_polity is them.previous_polity and us.previous_polity is not None
            if False and self.type is not 'land': # DEBUG
                cw_tag = "c" if cw  else ""
                print("%d %s %sw: %d %d -> %d" % (whm.this_year,self.type,cw_tag,us.id,them.id,us_alt.id))
            global report_annexations
            if report_annexations:
                whm.annexation_data.append([whm.this_year,cw,us.id,them.id,us_alt.id,power_us,power_them,victorious_dead,vanquished_dead,len(annex_territories)])
            if dump_stats: # DEBUG
                cw_tag = "!" if cw  else ""
                report_random_counter()
                print("%d %sX: %d %d %d %.1f %.1f %.1f %.1f %d %d" % (whm.this_year,cw_tag,us.id,them.id,us_alt.id,
                                                                      power_us,power_them,victorious_dead,vanquished_dead,
                                                                      self.us_territory.t_index,battle_i))
        else:
            pass # hill skirmish -- no one dies, no territory changes hands


def power_projection_factor(distance,s,Boulding=True):
    # Compute the power project factor given a distance and a supply support factor
    # distance is in some scaled unit, typically degrees (~regions) or 100km.
    # s is in units supply personnel per soldier per distance unit
    # Observation over s range from 1 to 3 (typical for our time period) is that
    #  power_projection_factor(d,s,Boulding=True) ~= power_projection_factor(d,s+0.5,Boulding=False)
    if Boulding:
        return 1/(1 + 1/s)**distance  # Boulding's 1962 direct, binomial solution
    else:
        return 1/np.exp(distance/s) # Turchin's 2003 approximation to Boulding

class PolityCommon(ComparablePolity,EliteDemographicPolity): # Interitance order is important
##f WHM:__repr__(...)
##f WHM:annex(...)
##f WHM:collapse(...)
##f Population:compute_migration(...)
##f WHM:find_annexable(...)
##f WHM:find_borders(...)
##f WHM:large_enough(...)
##f WHM:machiavelli(...)
##f WHM:make_quasi_state(...)
##f PopElites:migrate_population(...)
##f WHM:same_polity(...)
##f WHM:set_border_cache(...)
##f PopElites:update_arable_k(...)

    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
##s PopElites.py:EliteDemographicPolity:__init__
        territories = [] if territories is None else territories
##s Population.py:DemographicPolity:__init__
        territories = [] if territories is None else territories
        global whm
        # Assume it will be a full-fledged state
        # we reset Unoccupied and Hinterland in WHM.setup_for_trial() below
        # but then the rates are fixed for the entire simulation
        self.density = whm.state_density
        self.br = whm.state_birth_rate
        
        self.total_population = 0 
        self.max_population = 0 
        self.total_k = 1 # avoid DBZ in logistic eqn below

##s WHM.py:Polity:__init__
##s WHM.py:State:__init__
        global whm
        self.flag = next(whm.flag_generator)
        # DEAD self.name = name if name else hex(id(self))
        # TODO rather than using colors use a counter reset every time step?
        # '%s %s%s' % (whm.this_pretty_year, self.flag['marker'], self.flag['color'])
        # self.name = name if name else '%s 0x%x' % (whm.this_pretty_year, id(self))
        # Make default name reflect year of birth and a unique decimal id
        # Makes it easy to parse and read in matlab
        global polity_counter
        self.id = polity_counter
        self.name = name if name else '%d %d' % (whm.this_year, self.id) # id(self)
        polity_counter += 1


        # TODO consider a cache __regions__ of the extended regions that changes via Territory.set_polity()
        # initialize here given territories
        whm.polities.append(self) # unique and in order by definition
        self.previous_polity = None # previous 'parent' polity after a collapse (None if sui generus)
        self.large_enough_ignore = False # as a rule, count this polity in the statistics if large_enough()
        self.permit_resurrection = False # once you are dead you can't come back
        self.start_year = whm.this_year
        self.end_year = self.start_year
        # This list and counters are maintained by Territory.set_polity(), which see
        self.territories = set() if use_set_versions else list() # nothing yet
        # maintain a list of territory indicies for addressing parts of world-wide arrays such as population, k, etc.
        self.territories_i = [] # no indices yet
        self.agricultural_i = [] # the subset that is arable
        self.size = 0 # regions
        self.max_size = 0 # regions
        self.km2_size = 0 # km2
        self.quasi_state = False
        # NOTE: if no territories are specified here, some had best be added before the next call to state_of_the_union()
        # or this polity will be marked DEAD assuming annexation
        self.set_state(POLITY_ALIVE)  # do this last so quasi_state is set
        self.set_border_cache() # clear border cache for political and physical on polity

    # Things like Hinterland and Unoccupied are basically individual territories that share properties
    # but are not a collective like a polity.  Each territory is independent.  Quasi states never 'die'
##e WHM.py:State:__init__
##< WHM.py:Polity:__init__

        territories = [] if territories is None else territories # supply mutable default value
        self.is_confederation = False
        self.confederation = None
        # UNNEEDED since set_polity() will clear cache each time
        # self.set_border_cache() # initialize border cache for political and physical as empty
        for territory in territories:
            territory.set_polity(self)
        
##e WHM.py:Polity:__init__
##< Population.py:DemographicPolity:__init__


##e Population.py:DemographicPolity:__init__
##< PopElites.py:EliteDemographicPolity:__init__

        global whm
        self.elite_k_fraction = whm.base_elite_k_fraction

    # This should be called whenever the set of territories changes
##e PopElites.py:EliteDemographicPolity:__init__
        global whm
        self.polity_type = POLITY_TYPE_AG # typically
        # nomad support but set regardless of enabled or not
        self.extorting = [] # if confederation or tribe, set of agrarian states it is extorting; empty if agrarian
        self.extorted_by = [] # if agrarian, which confederations or tribes are pestering it; empty if nomadic
        self.recompute_power_scale = False

        # where is the capital? 
        self.Clon = np.nan
        self.Clat = np.nan
        self.starting_size = len(self.territories) # for cycle analysis
        self.max_size = self.starting_size
        self.max_territories_i = copy.copy(self.territories_i) # initial values
        self.max_year = self.start_year
        self.succession_crisis_year = whm.this_year + whm.succession_generation_time
        self.continuations = 0 # vestigial
        if self.starting_size:
            self.update_capital()

    def update_capital(self,core_polity=None):
        global whm,geo
        if self.quasi_state: # these don't have capitals
            return False
        territories_i = self.territories_i
        if len(territories_i) > 0 and self.Clon is np.nan:
            Clat = int(np.mean(whm.latitude[territories_i]))
            Clon = int(np.mean(whm.longitude[territories_i]))
            # Has supply depot location changed?
            if Clon is not self.Clon or Clat is not self.Clat:
                # original version
                self.Clon = Clon
                self.Clat = Clat
                self.update_distance_from_capital()
                return True
        return False

    def update_distance_from_capital(self):
        global whm,geo

        if self.Clon is np.nan: # brand new polity?
            self.update_capital() # this calls us recursively
            return
            
        wlat = whm.latitude
        wlon = whm.longitude

        # scale distance by the mean latitude of a 'region' centered at the depot
        lat_scale = np.cos(np.radians((wlat + self.Clat)/2.0))
        lat_km = geo.lat_km_per_degree
        lon_km = geo.lon_km_per_degree
        distance_scale = whm.km_distance_scale
        dlat = wlat - self.Clat
        dlon = wlon - self.Clon
        # use of these routines doubles the time of an unfold?
        # These sea/desert calculations only need to be done once any of the sea_connections_i or desert_connections_i are annexed by the state itself
        max_distance_from_capital_km = horse_cavalry_diffusion_distance # (120.0/3.0)*111.32*np.cos(np.radians(30))
        # TODO Calls to this function are the main expense of a run (60%!)
        # Anything we can do to reduce this cost would be permit more runs!!
        # If we knew that the factors are fixed we could compute and cache a distance array for each ag region
        # that could ever be a capital.
        # We call this every time a capital is formed (new state) and for every POLITY_ALIVE state whenever an efficiency factor changes
        def update_d(distance_from_capital,embarkation_i,factor):
            # OPTIMIZATION: no state will get big enough to reach certain embarkation points
            # (e.g., China will never make it to the ME or the Med)
            # don't bother computing discounted paths from them
            close_embarkation_i = np.where(distance_from_capital[embarkation_i] <= max_distance_from_capital_km)[0]
            for e_i in embarkation_i[close_embarkation_i]:
                # compute the distance from each embarkation point to all others
                # The demb calculation could be cached regardless of factor but it would be len(embarkation_i)^2 in space
                elon = wlon[embarkation_i] - wlon[e_i]
                elat = wlat[embarkation_i] - wlat[e_i]
                demb = np.sqrt((lat_scale[embarkation_i]*elat*lat_km)**2 + (elon*lon_km)**2)
                dem = distance_from_capital[e_i] + factor*demb # distance from the capital to the embarkation point e_i and then to all other disembarkation points
                ddiff = dem - distance_from_capital[embarkation_i] # difference going by this means of travel (via factor) vs direct distance
                dd_i = np.nonzero(ddiff < 0)[0] # where is it shorter to go by this means of travel?
                ds_i = embarkation_i[dd_i] # use these embarkation points
                distance_from_capital[ds_i] = dem[dd_i] # and this distance (update since we know these are more efficient via embarkation)
                for ds_ii in ds_i: # for each point recompute the distance from disembarkation point to final locations
                    # this too could be cached but it is a full set of distances for each embarkation point so memory intensive
                    xlon = wlon - wlon[ds_ii]
                    xlat = wlat - wlat[ds_ii]
                    dsx = np.sqrt((lat_scale*xlat*lat_km)**2 + (xlon*lon_km)**2)
                    dxx = distance_from_capital[ds_ii] + dsx
                    ddiff = dxx - distance_from_capital
                    dd_i = np.nonzero(ddiff < 0)[0]
                    distance_from_capital[dd_i] = dxx[dd_i]
            return distance_from_capital

        # NOTE: when computing distances, since we are usually within a nearby patch
        # the Euclidean distance suffices as long neither dlat nor dlon exceed 180 degrees
        distance_from_capital = np.sqrt((lat_scale*dlat*lat_km)**2 + (dlon*lon_km)**2) # Euclidean km
        if whm.sea_distance_factor and len(geo.sea_connections_i):
            distance_from_capital = update_d(distance_from_capital,geo.sea_connections_i,whm.sea_distance_factor)
        if whm.desert_distance_factor and len(geo.desert_connections_i):
            distance_from_capital = update_d(distance_from_capital,geo.desert_connections_i,whm.desert_distance_factor)
        self.distance_from_capital = distance_from_capital/distance_scale
        

    def marked_polity(self):
        global whm
        marked_polities = whm.display_marked_polities
        if marked_polities is None:
            return False
        return (len(marked_polities) == 0 or self.id in marked_polities)

    # Only called if POLITY_ALIVE
    def state_of_the_union(self):
##s ComparativeHx.py:ComparablePolity:state_of_the_union
##s Population.py:DemographicPolity:state_of_the_union
##s WHM.py:Polity:state_of_the_union
        # only called when POLITY_ALIVE
        global whm
        self.end_year = whm.this_year # CRITICAL update to latest year alive
    

##e WHM.py:Polity:state_of_the_union
##< Population.py:DemographicPolity:state_of_the_union


        # CONSTRAINT udpate total_population here
        # NOTE the chronicler wants it for display during state_of_the_world()

        if self.size > 0: # member or alive: and self.state is POLITY_ALIVE:
            # Migration and total_pop is a per-polity activity
            
            # We migrate extant people in live polities and compute
            # total population here.  We grow people en-mass during state_of_the_world()

            self.update_arable_k()
            global whm
            if self.quasi_state:
                pass # we don't migrate quasi_states
            else:
                self.migrate_population(migration_factor=whm.migration_fraction)
            t_i = self.territories_i
            self.total_population = sum(whm.population[t_i])
            if self.total_population > self.max_population:
                self.max_population = self.total_population
            self.total_k = sum(whm.k[t_i])


    # This should be called whenever the set of territories changes
##e Population.py:DemographicPolity:state_of_the_union
##< ComparativeHx.py:ComparablePolity:state_of_the_union

        # Always collect the prediction data and reset below
        # define the criteria for counting this polity
        # We want it to be large enough but there is no restrictions on states being at least a century old
        # Thus we count it as a valid prediction even if there are several large_enough() states during the century
        if self.state == POLITY_ALIVE and self.large_enough(): # avoid counting any POLITY_MEMBERs
            global whm
            # TODO build a global index of territories in track_polity_biomes and update it whenever/if a territory changes biomes
            # then replace this calculation of regions with an np.intersect expression
            regions = []
            for territory in self.territories:
                if territory.Biome in whm.track_polity_biomes:
                    regions.extend(territory.regions)
            whm.predictions[regions] += 1
            global dump_predicted_history
            if dump_predicted_history:
                try:
                    int_this_year = int(whm.this_year)
                    century_i = whm.geography.years.index(int_this_year)
                except (ValueError, IndexError ):
                    pass
                else:
                    # BUG This records the id of the state at the century mark, not whether it has lasted a century
                    whm.predicted_history[regions,century_i] = self.id

# Mixing with WHM
##e ComparativeHx.py:ComparablePolity:state_of_the_union
        # state and population etc all updated
        # did state remain alive?
        if self.state is POLITY_ALIVE:
            global whm
            # record the maximum extent of this polity and the year it happened
            if self.size == self.max_size:
                self.max_territories_i = copy.copy(self.territories_i) # toplevel copy
                self.max_year = whm.this_year


    # maintain the extortion relationship between nomadic and agrarian polities
    def extort(nomadic_self,agrarian_polity,no_longer=False):
        if no_longer:
            # mark that polity is no longer being extorted by self
            nomadic_self.extorting.remove(agrarian_polity)
            agrarian_polity.extorted_by.remove(nomadic_self)
        else:
            # mark that polity is now being extorted by self
            add_unique(nomadic_self.extorting,agrarian_polity)
            add_unique(agrarian_polity.extorted_by,nomadic_self)

    def set_state(self,state):
##s Population.py:DemographicPolity:set_state
##s WHM.py:State:set_state
        if not self.quasi_state: # We never change the state of quasi_states
            self.state = state

##e WHM.py:State:set_state
##< Population.py:DemographicPolity:set_state

        if self.state not in [POLITY_ALIVE, POLITY_MEMBER]:
            # Ensure that DEAD polities have no (total_)population
            # or you will get bad world_population counts
            self.total_population = 0
            # do not reset max_population
            self.total_k = 1 # POLITY_DEAD but avoid DBZ in logistic calculation below
            
    
##e Population.py:DemographicPolity:set_state
        if self.state is not POLITY_ALIVE:
            for nomadic_polity in copy.copy(self.extorted_by):
                nomadic_polity.extort(self,no_longer=True)  # polity can no longer pester self
            for agrarian_polity in copy.copy(self.extorting):
                self.extort(agrarian_polity,no_longer=True) # self can no longer pester polity
            # reset (though they should both be empty by now)
            self.extorting = []
            self.extorted_by = []
                
    def ag_polities_within_striking_distance(nomadic_self):
        global whm
        # If self is an NC, then this cached list reflects the union of all the ag_territories_striking_distance_NC territories on all members
        striking_distance_territories = nomadic_self.ag_territories_striking_distance_NT
        ag_within_striking_distance = unique([t.polity for t in striking_distance_territories])
        ag_within_reach = []
        max_s_polity = None
        max_pop_polity = None
        total_extortable_population = 0
        for ag_polity in ag_within_striking_distance:
            if ag_polity == whm.Hinterland:
                continue # skip extorting Hinterland
            territories = ag_polity.territories
            striking_territories = intersect(territories,striking_distance_territories)
            # could this tribe/confed pester this polity enough?
            fraction_within_reach = float(len(striking_territories))/float(len(territories))
            if fraction_within_reach < whm.fraction_ag_state_terriorties_within_strike_distance:
                continue

            ag_within_reach.append(ag_polity)
            total_extortable_population += ag_polity.total_population
            if max_s_polity is None:
                max_s_polity = ag_polity
                max_pop_polity = ag_polity
            else:
                if ag_polity.s > max_s_polity.s:
                    max_s_polity = ag_polity
                if ag_polity.total_population > max_pop_polity.total_population:
                    max_pop_polity = ag_polity
        return (ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity)

# ---- agrarian support ----
class AgrarianPolity(PolityCommon):
##f WHM:__repr__(...)
##f WHM:annex(...)
##f WHM:collapse(...)
##f Population:compute_migration(...)
##f WHM:find_annexable(...)
##f WHM:find_borders(...)
##f WHM:large_enough(...)
##f WHM:make_quasi_state(...)
##f PopElites:migrate_population(...)
##f WHM:same_polity(...)
##f WHM:set_border_cache(...)
##f PopElites:update_arable_k(...)
    # Maintains a capital, fixed or not
    # Maintains a power array from capital to all (distant) territories
    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
        global whm
        # how does power scale to all distant territories PER POLITY
        self.s = whm.s # Could involve Kremer
        # these vector caches both get reset
        self.power_scale = np.zeros(len(whm.territories),dtype=float )
        self.distance_from_capital = self.power_scale
        self.year_of_normal_tolerance = whm.this_year # when do we reset to base_misery_threshold? (NOW basically)
        self.misery_threshold = whm.base_misery_threshold
        # This call will update territories, distance_from_capital etc.
## super calls PolityCommon:__init__(...)
        super().__init__(name=name,territories=territories)
        

    def extort(self,polity):
        raise RuntimeError("How can agrarian polity %s be extorting?" % self)

    def set_state(self,state):
## super calls PolityCommon:set_state(...)
        super().set_state(state)
        if self.state not in [POLITY_ALIVE,POLITY_MEMBER]:
            # clear these caches to reduce memory footprint
            self.power_scale = None
            self.distance_from_capital = None

    def collapse_to_hinterland(self,reason=''):
        global whm
        if self.size > 3: # reduce chatter
            print("Collapsing %s to %d hinterland in %s%s!" % (self,self.size,whm.this_pretty_year,reason)) #DEBUG

        #DEBUG print 'Collapsing: %s(%dy %d/%d)' % (self,whm.this_year - self.start_year,self.size,self.max_size) # DEBUG
        # Collapse this polity back to the stone age
        # It will always be to Hinterland and not Unoccupied
        # since we only permit states in GEO_AGRICULTURAL
        Hinterland = whm.Hinterland
        # since self.territories is (sometimes) a set and the call to set_polity()
        # modifies the set during iteration, the iterator fails with
        # RuntimeError, "Set changed size during iteration"
        # protect against that by copying to a list for the iterator
        # Done once per collapsing polity so no big deal
        for territory in copy.copy(self.territories):
            # ALTERNATIVE territory.set_polity(territory.quasi_state)
            territory.set_polity(Hinterland)
            
        # Let the normal course of events update self via state_of_the_union()
        self.set_state(POLITY_COLLAPSED)

    def create_factions(self,ph_t,n_factions):
        global whm
        ph_t.sort(key=lambda t:t.t_index) # put into canonical order

        collapse_factions = whm.collapse_factions # get our helpful create faction array
        collapse_factions[:] = UNAVAILABLE # reset the world to unavailable
        collapse_factions[self.territories_i] = AVAILABLE # available

        # precompute all the combined land, sea, and desert connections for each territory in the collapsing polity
        connections = {}
        for territory in self.territories:
            connections[territory] = territory.get_connections(sea=whm.sea_battle_distance,desert=whm.desert_battle_distance,combined=True)
        
        factions = []
        for seed_t in urandom_sample(ph_t,n_factions):
            factions.append(([seed_t],[]))
            collapse_factions[seed_t.t_index] = ASSIGNED # assigned to a faction
        some_faction_expandable = True # does any faction still have the ability to expand?
        while some_faction_expandable:
            some_faction_expandable = False
            for available,filled in factions:
                if len(available):
                    this_faction_expandable = False
                    # copy.copy since we might be removing elements below
                    for a_t in copy.copy(available):
                        aa_t = None
                        # TODO if we permit sea connections, look at land and sea connections within striking distance 
                        for c_t in connections[a_t]:
                            if c_t.littoral:
                                continue # skip ocean markers
                            if collapse_factions[c_t.t_index] == AVAILABLE: # available?
                                # Setting this true each faction has 1 chance at something available per 'round'
                                # and ensures still might have a chance on the next round, regardless of whether the territory joins them
                                this_faction_expandable = True 
                                # decide whether this one joins the faction
                                if urandom_random() < whm.faction_add_odds:
                                    aa_t = c_t
                                break
                        if this_faction_expandable:
                            some_faction_expandable = True # go another round
                            if aa_t:
                                available.append(aa_t)
                                # now unavailable
                                collapse_factions[aa_t.t_index] = ASSIGNED # assigned to a faction
                            break # move to next faction
                        else:
                            filled.append(a_t)
                            # This is ok because we copy above
                            available.remove(a_t) 
                            # keep looking for more available
        remaining_territories_i = np.nonzero(collapse_factions == AVAILABLE)[0]
        return (factions,remaining_territories_i,)

    def collapse_to_factions(self,why):
        global whm
        initial_size = self.size
        n_factions = urandom_choice(whm.faction_odds)
        po_t,ph_t = self.find_borders(sea=whm.sea_battle_distance,desert=whm.desert_battle_distance,physical=True)
        ph_t = list(set(ph_t) | set(po_t)) # union(ph_t,po_t)
        # If the current polity would collapse into tiny factions
        # or we can't find enough border seeds, collapse to hinterland
        if initial_size < 2*n_factions or len(ph_t) < n_factions:
            return self.collapse_to_hinterland()
        if n_factions > 2: # DEBUG
            print('Collapsing: %s(%dy %d/%d) into %d factions' % (self,whm.this_year - self.start_year,self.size,self.max_size,n_factions))
        # create_factions puts ph_t into canonical order
        (factions,remaining_territories_i) = self.create_factions(ph_t,n_factions)

        try:
            hx_entry = whm.collapse_history[self]
        except KeyError:
            hx_entry = [self,[]]
            whm.collapse_roots.append(self)
            whm.collapse_history[self] = hx_entry # intern
        hx_entry = hx_entry[1] # append to this

        total_deaths = 0;
        n_factions = 0 # reset and count actual factions formed (vs. Hinterland)
        n_hinterland = 0 # count of territories that fell back to hinterland
        Hinterland = whm.Hinterland
        faction_states = []
        for available,filled in factions:
            if len(filled) == 1:
                # Single regions are Hinterland by definition
                n_hinterland += 1
                # ALTERNATIVE filled[0].set_polity(filled[0].quasi_state)
                filled[0].set_polity(Hinterland)
            else:
                # New state arises from the ashes
                faction = whm.PolityType(territories=filled)
                faction_states.append(faction)
                faction.previous_polity = self
                faction_hx = [faction,[]]
                whm.collapse_history[faction] = faction_hx # intern
                hx_entry.append(faction_hx)
                n_factions += 1


        n_hinterland += len(remaining_territories_i)
        if n_hinterland:
            # why did we miss thse guys?
            for rm_i in remaining_territories_i:
                territory = whm.territories[rm_i]
                # ALTERNATIVE territory.set_polity(territory.quasi_state)
                territory.set_polity(Hinterland)

        def prepare_for_cw(faction):
                # starts with at least the parent's s but in the case of Kremer s
                # ensure it has a chance as well
                faction.s = max(self.s,faction.s) 
                faction.s = whm.bounded_s(faction,0,whm.max_polity_s)
                # leave the extorted field alone
                faction.update_power()

                # Calling state_of_the_union() here finds the new polity has
                # political boundaries against a mix of the old state (self) and
                # possibly some of the new factions.  But that is ok since we
                # just care to locate the internal boundaries to waste people

                filled_i = faction.territories_i
                # How many elites per region are over- or under- new minimum misery tolerance fraction?
                deaths = 0
                residual_elites = (whm.elites[filled_i] -
                                   whm.collapse_elite_population_fraction*whm.k[filled_i]*whm.elite_k_fraction[filled_i])
                too_many_i = np.nonzero(residual_elites > 0)[0]
                if len(too_many_i):
                    deaths = np.sum(residual_elites[too_many_i]) # record them
                    whm.elites[np.array(filled_i)[too_many_i]] -= residual_elites[too_many_i] # remove them

                # inlined elite_opportunity calculation here since we are resetting variables directly
                elite_k = whm.k[filled_i]*whm.elite_k_fraction[filled_i] # what about regional_intensification??
                whm.elite_opportunity[filled_i] = (1 - whm.elites[filled_i]/elite_k)

                # In addition, potentially increase the faction's misery threshold for the required amount of time
                faction.misery_threshold = whm.increased_misery_threshold
                # But we are tolerant for only so long...
                # This handles the case where the faction actually survives and acquires enough opportunity to avoid
                # another collapse and so we return to normal.  If we don't we will collapse in the year computed below anyway
                faction.year_of_normal_tolerance = whm.this_year + whm.faction_increased_asabiya_time
                # update its raw_power, capital, etc, with whatever elites remain
                faction.state_of_the_union()
                return deaths

        for faction in faction_states:
            total_deaths += prepare_for_cw(faction)

        self.set_state(POLITY_COLLAPSED)

        global report_polity_history
        if report_polity_history:
            whm.collapse_data.append([whm.this_year,self.id,whm.this_year - self.start_year,n_factions,n_hinterland,total_deaths])
        # it is possible for initial_elites to be zero in which case total_deaths*100/initial_elites is nan; eliminate from print
        if True and initial_size >= whm.compare_polity_size: # DEBUG reduce chatter
            report = "Collapsed %s (%d regions) in %s after %d years to %d factions because of %s; %d hinterland territories; %.1f deaths"
            print(report % (self,initial_size,whm.this_pretty_year,whm.this_year - self.start_year, n_factions,why,n_hinterland,total_deaths)) 

        for faction in faction_states:            
            if urandom_random() < whm.faction_collapse_odds:
                faction.collapse_to_factions('after faction formation') # Note that parent is the faction, not self!!

    def state_of_the_union(self):
        global whm
        # NOTE if we are a member we won't be ALIVE 
        if not self.quasi_state and self.size and self.state is POLITY_ALIVE: # alive with territories
            if self.year_of_normal_tolerance <= whm.this_year:
                # Reset to normal misery tolerance
                self.misery_threshold = whm.base_misery_threshold

            if self.size == 1: 
                self.collapse_to_hinterland() # collapse to hinterland
                return
            # Controlled by the number of fraction of territories containing miserable elite entourages
            miserable_elite_territories = np.nonzero(whm.elite_opportunity[self.territories_i] < (1 - self.misery_threshold))
            # Simple rule: Any territory that goes miserable for elites triggers collapse
            # Given the rapidity with which territories are added there are typically
            # several that are right behind the initial core group
            # Since self.state could be POLITY_COLLAPSED we guard all subsequent code on POLITY_ALIVE
            if len(miserable_elite_territories[0]):
                self.collapse_to_factions("misery")
                # FALLTHROUGH

## super calls PolityCommon:state_of_the_union(...)
        super().state_of_the_union()

        recompute_power_scale = self.recompute_power_scale
        if not self.quasi_state and self.size and self.state is POLITY_ALIVE: # still alive with territories
            # COMMAND AND CONTROL: compute any improvements in scale of force or improved projection
            if whm.s_adapt_factor:
                po_t,ph_t = self.find_borders(sea=whm.sea_battle_distance,desert=whm.desert_battle_distance,physical=True)
                annexable_t = self.find_annexable(po_t=po_t)

                border_s = self.s
                s_adapt_factor = whm.s_adapt_factor;
                if len(annexable_t):
                    max_s = max( [t.polity.s for t in annexable_t])
                    border_s = max(max_s,border_s)
                if len(self.extorted_by):
                    max_s = max([n.s for n in self.extorted_by])
                    if max_s > border_s:
                        border_s = max_s
                change_s = s_adapt_factor*(border_s - self.s) # improve at a fraction of the difference
                # TODO? add some gaussian stochastic variance there?  e.g., random.normal(0,-0.01)
                if change_s > 0: # ensure we only ratchet s up, no weakening if pressure goes away
                    self.s = whm.bounded_s(self,change_s,whm.max_polity_s)
                    recompute_power_scale = True

        # state and population etc all updated
        # did state remain alive?
        if self.state is POLITY_ALIVE:
            self.update_power(force_recompute=recompute_power_scale)

    def update_power(self,force_recompute=True):
        # Assumes self is POLITY_ALIVE

        # For the most part Unoccupied and Hinterland can avoid this since their power is defined by raw_territorial_power
        # We need to compute the (political) borders at least, even for Hinterland, since that is where wars are ordered
        # But for true states we need to compute both political and physical boundaries so we can scale raw_power properly
        
        # cells on borders of other polities (including Hinterland against some other Polity)
        # border asabiya grows under constant threat from 'others'
        global whm
        territories_i = self.territories_i
        # Find the political and physical borders
        po_t,ph_t = self.find_borders(sea=whm.sea_battle_distance,desert=whm.desert_battle_distance,physical=True)
        self.border_length = len(po_t) + len(ph_t)
        # ADSM do we scale non-political per Artzrouni?
        # self.border_cost = max(1.00*len(po_t) + 0.95*len(ph_t)) # Artzrouni values
        # possible for a pair of ag states to be surrounded by only steppe and water (eastern Black Sea, around 42E/42N)
        self.border_cost = max(1.0*len(po_t) + 0.0*len(ph_t),1)
        # Cache only the political borders for War generation
        self.border_territories = po_t
        border_territories_i = [t.t_index for t in po_t] 
        self.border_territories_i = border_territories_i # save this for debugging

        # don't need to do this for Hinterland and Unoccupied
        if self.quasi_state:
            self.raw_power = 0
        else:
            self.raw_power  = sum(whm.elites[territories_i]) 
            self.raw_power /= self.border_cost

        # TODO since we cache borders, if this cache is present then nothing has changed so we don't need to recompute
        # BUT you need to record that fact before you call find_borders() above!!
        # power_scale can change if either the capital changes location or s changes
        # in both cases, we recompute the distance from capital, even if just changing s,
        # which is expensive in time but caching is expensive in space
        # By default we recompute the power_scale but during state_of_the_union() if we haven't changed s or the capital
        # then the projection function values don't change so we won't force_recompute
        recompute_power_scale = force_recompute
        if self.update_capital():
            recompute_power_scale = True

        if recompute_power_scale:
            if not getattr(self,'s',False):
                raise RuntimeError("How can %s not have s set?" % self)
            self.power_scale = power_projection_factor(self.distance_from_capital,self.s) # HEGEMON was whm.s ADSM eqn 7
            self.recompute_power_scale = False

        # DEBUG -- display various stats about each ALIVE state
        if dump_stats and not self.quasi_state:
            ph_t = union(ph_t,po_t) # combine them using our version....
            border_territories_i = [t.t_index for t in ph_t] # canonical order does not matter here
            most_distant_border_i = np.argmin(self.power_scale[border_territories_i])
            most_distant_border_i = border_territories_i[most_distant_border_i]
            power_scale = self.power_scale[most_distant_border_i]
            PE_format = "%d %d PE: %d %d %d %.4f %.1f %.1f %.1f %.1f %.1f %.4f %.1f %.2f"
            print(PE_format % (whm.this_year,self.id,len(territories_i),
                               self.border_length,self.border_cost,
                               np.min(whm.elite_opportunity[self.territories_i]),
                               self.total_k,self.total_population,
                               sum(whm.peasants[territories_i]),
                               sum(whm.elites[territories_i]),
                               self.raw_power,power_scale, 
                               self.raw_power*power_scale, # army size
                               self.s)) 
            
    def machiavelli(self):
        global whm
        orders = []
        Unoccupied = whm.Unoccupied
        Hinterland = whm.Hinterland
        if self is Unoccupied:
            return orders

        if self is not Hinterland:
            if self.confederation is None:
                # JOIN: return if yes
                pass
            else:
                # SECEDE: return if yes
                pass

        # ANNEXATION:
        if self is Hinterland:
            # all hinterland can battle themselves
            # but, for speed, we prune orders below for the majority that don't have a power differential
            border_territories = self.territories 
        else:
            border_territories = self.border_territories
        border_territories = list(border_territories)
        min_intensification = whm.War_min_intensification
        max_elevation = whm.War_max_elevation

        attack_entries = {}

        for us_territory in border_territories:
            lc,sc,dc = us_territory.get_connections(sea=whm.sea_battle_distance,desert=whm.desert_battle_distance,combined=False)
            for them_territory in lc:
                war_type = 'land';
                if them_territory.littoral: # Crossing the sea?
                    if not len(sc):
                        continue;
                    war_type = 'sea'
                    them_territory = urandom_choice(sc)
                else:
                    if whm.geography.Biome[them_territory.t_index] == GEO_DESERT:# Crossing the desert?
                        if not len(dc):
                            continue;
                        war_type = 'desert'
                        them_territory = urandom_choice(dc)
                # at this point we have a potential 'border' territory
                them_polity = them_territory.polity
                # skip unoccupied or nomadic polities
                if  them_polity is Unoccupied or not isinstance(them_polity,AgrarianPolity):
                    # Nothing to gain here
                    continue
                if them_polity is self:
                    # same polity
                    if self is Hinterland:
                        if avoid_hinterland_hinterland_annexation:
                            continue
                        # We have a possible H:H battle; issue an order only if there is a gradient in the right direction
                        # NOTE: we need to compare ratio of powers here and because of the different sizes of regions by latitude
                        # a more region closer to the equator will have slightly more people (hence power) than a region just one step
                        # toward the pole (in whatever hemisphere).  This difference will vary with latitude but at worst (near the poles)
                        # it is about 2% so that is the floor for the power excess.  If it is less than that, we will have a wave of hinterland
                        # states forming at some mid-latitude, moving north and then south...
                        if (whm.raw_territorial_power[  us_territory.t_index]*whm.power_scale_1[us_territory.lat] <
                            whm.raw_territorial_power[them_territory.t_index]):
                            continue
                        #DEBUG print 'HH' # DEBUG
                    else:
                        # not Hinterland: can't attack self
                        continue
                    # else different polities but could be hopelessly outgunned
                try:
                    attack_entries[them_territory] # already seen?
                except KeyError:
                    attack_entries[them_territory] = (us_territory,war_type,them_polity)

        # We now have a set of all (unique) territories to attack for this polity
        attack_territories = list(attack_entries.keys()) # the keys are in memory address order so even though it is a list it is a 'set'
        attack_territories.sort(key=lambda t:t.t_index) # order canonically by index
        
        # now sort by preference, if any
        attack_territories = urandom_sample(attack_territories,len(attack_territories)) # original algorithm: random shuffle
        # choose some top fraction
        n_at = float (len(attack_entries));
        a_at = int(n_at*whm.attack_budget_fraction)
        attack_territories = attack_territories[0:a_at];
        # filter those and record remaining orders
        for them_territory in attack_territories:
            if whm.intensification_factor[them_territory.t_index] < min_intensification:
                continue # Not worth the effort
            if them_territory.Elevation >= max_elevation:
                continue # Not worth the effort
            us_territory,war_type,them_polity = attack_entries[them_territory]
            war = War(us_territory,them_territory)
            war.type = war_type
            orders.append(war)
        # print "ORDERS: %d %d" % (len(orders),len(border_territories)) # DEBUG STATISTICS
        return orders

# ---- nomads --- 

# how nomadic tribes officially join nomadic confederations
# they depart via confederation collapse
class JoinNomadicConfederation(Order):
    def __init__(self,tribe,confederation,type):
        self.tribe = tribe
        self.confederation = confederation # record the confederation we are joining (different from is_confederation)
        self.type = type

    def execute(self):
        global whm
        if (self.confederation.state is POLITY_ALIVE and # ensured?
            self.tribe.state is POLITY_ALIVE):
            tribe = self.tribe
            confederation = self.confederation
            if tribe.confederation is not None:
                if False: # we had option and joined another confederation first? How can this be since we only choose one confed per mach() round?
                    # Because we both raise an NC with it (but not by it!)
                    # but in the same round it hasn't been converted to annexed so we also choose a different confed to join
                    raise RuntimeError('How can this be? %s already part of %s but now going to %s?' % (tribe,tribe.confederation,confederation))
                return
            # Testing on first_confederation cuts down chatter but
            # can make it look like no NC adds tribes over time
            if tribe.first_confederation is None:
                # print('Adding tribe %s to %s' % (tribe,confederation))
                tribe.first_confederation = confederation
            if True: # DEBUG
                print('%s: Adding %s tribe %s to %s' % (whm.this_pretty_year,self.type,tribe,confederation))
            confederation.join(tribe)
            confederation.s = max(tribe.s,confederation.s) # at least

            # update increaed K here? No let the NC do that in SOU()
            # NOTE: here we update the NC polity's ag_territories_striking_distance_NT from the cached NC regions
            ag_territories = []
            for tribe in confederation.member_polities:
                ag_territories.extend(tribe.ag_territories_striking_distance_NC) # could be more distant
            confederation.ag_territories_striking_distance_NT = unique(ag_territories)

# nomadic polities are either one of a set of fixed (random per run) NomadicTribe
# or collections of (territories) from a set of tribes NomadicConfederation
class NomadicConfederation(Confederation,PolityCommon): # inheritance order is key
##f WHM:__repr__(...)
##f WHM:annex(...)
##f WHM:assert_core_polities(...)
##f WHM:collapse(...)
##f Population:compute_migration(...)
##f WHM:find_annexable(...)
##f WHM:find_borders(...)
##f WHM:join(...)
##f WHM:large_enough(...)
##f WHM:make_quasi_state(...)
##f PopElites:migrate_population(...)
##f WHM:same_polity(...)
##f WHM:set_border_cache(...)
##f WHM:split(...)
##f PopElites:update_arable_k(...)
    def __init__(self,name=None):
        global whm
##s WHM.py:Confederation:__init__
##s NADSM.py:PolityCommon:__init__
        territories = [] if territories is None else territories
##s PopElites.py:EliteDemographicPolity:__init__
        territories = [] if territories is None else territories
##s Population.py:DemographicPolity:__init__
        territories = [] if territories is None else territories
        global whm
        # Assume it will be a full-fledged state
        # we reset Unoccupied and Hinterland in WHM.setup_for_trial() below
        # but then the rates are fixed for the entire simulation
        self.density = whm.state_density
        self.br = whm.state_birth_rate
        
        self.total_population = 0 
        self.max_population = 0 
        self.total_k = 1 # avoid DBZ in logistic eqn below

##s WHM.py:Polity:__init__
##s WHM.py:State:__init__
        global whm
        self.flag = next(whm.flag_generator)
        # DEAD self.name = name if name else hex(id(self))
        # TODO rather than using colors use a counter reset every time step?
        # '%s %s%s' % (whm.this_pretty_year, self.flag['marker'], self.flag['color'])
        # self.name = name if name else '%s 0x%x' % (whm.this_pretty_year, id(self))
        # Make default name reflect year of birth and a unique decimal id
        # Makes it easy to parse and read in matlab
        global polity_counter
        self.id = polity_counter
        self.name = name if name else '%d %d' % (whm.this_year, self.id) # id(self)
        polity_counter += 1


        # TODO consider a cache __regions__ of the extended regions that changes via Territory.set_polity()
        # initialize here given territories
        whm.polities.append(self) # unique and in order by definition
        self.previous_polity = None # previous 'parent' polity after a collapse (None if sui generus)
        self.large_enough_ignore = False # as a rule, count this polity in the statistics if large_enough()
        self.permit_resurrection = False # once you are dead you can't come back
        self.start_year = whm.this_year
        self.end_year = self.start_year
        # This list and counters are maintained by Territory.set_polity(), which see
        self.territories = set() if use_set_versions else list() # nothing yet
        # maintain a list of territory indicies for addressing parts of world-wide arrays such as population, k, etc.
        self.territories_i = [] # no indices yet
        self.agricultural_i = [] # the subset that is arable
        self.size = 0 # regions
        self.max_size = 0 # regions
        self.km2_size = 0 # km2
        self.quasi_state = False
        # NOTE: if no territories are specified here, some had best be added before the next call to state_of_the_union()
        # or this polity will be marked DEAD assuming annexation
        self.set_state(POLITY_ALIVE)  # do this last so quasi_state is set
        self.set_border_cache() # clear border cache for political and physical on polity

    # Things like Hinterland and Unoccupied are basically individual territories that share properties
    # but are not a collective like a polity.  Each territory is independent.  Quasi states never 'die'
##e WHM.py:State:__init__
##< WHM.py:Polity:__init__

        territories = [] if territories is None else territories # supply mutable default value
        self.is_confederation = False
        self.confederation = None
        # UNNEEDED since set_polity() will clear cache each time
        # self.set_border_cache() # initialize border cache for political and physical as empty
        for territory in territories:
            territory.set_polity(self)
        
##e WHM.py:Polity:__init__
##< Population.py:DemographicPolity:__init__


##e Population.py:DemographicPolity:__init__
##< PopElites.py:EliteDemographicPolity:__init__

        global whm
        self.elite_k_fraction = whm.base_elite_k_fraction

    # This should be called whenever the set of territories changes
##e PopElites.py:EliteDemographicPolity:__init__
##< NADSM.py:PolityCommon:__init__

        global whm
        self.polity_type = POLITY_TYPE_AG # typically
        # nomad support but set regardless of enabled or not
        self.extorting = [] # if confederation or tribe, set of agrarian states it is extorting; empty if agrarian
        self.extorted_by = [] # if agrarian, which confederations or tribes are pestering it; empty if nomadic
        self.recompute_power_scale = False

        # where is the capital? 
        self.Clon = np.nan
        self.Clat = np.nan
        self.starting_size = len(self.territories) # for cycle analysis
        self.max_size = self.starting_size
        self.max_territories_i = copy.copy(self.territories_i) # initial values
        self.max_year = self.start_year
        self.succession_crisis_year = whm.this_year + whm.succession_generation_time
        self.continuations = 0 # vestigial
        if self.starting_size:
            self.update_capital()

##e NADSM.py:PolityCommon:__init__
##< WHM.py:Confederation:__init__

        self.is_confederation = True
        self.core_polity = None
        self.member_polities = [] # join and split manage this list
        
##e WHM.py:Confederation:__init__
        self.name = 'NC %s' % self.name
        self.polity_type = POLITY_TYPE_NC
        
        self.ag_territories_striking_distance_NT = [] # updated by join()
        self.max_territories_permitted = whm.confed_min_size
        self.s = whm.min_polity_s # (updated in calling code, JoinNomadicConfederation.execute() or NomadicConfederation:SOU())
        # the creator, below, executes a JoinNomadicConfederation order to update the tribes, territories, etc.
        # it also asserts the initial ag polity extorted
        # See NT:SOU to print NC Arising debug statement
        
 
    def collapse_to_tribes(self,reason='??'):
        global whm
        if True: # DEBUG
            print("Collapsing %s back to %d tribes after %d years due to %s!" % (self,len(self.member_polities),whm.this_year - self.start_year,reason)) #DEBUG
        # for tribe in self.member_polities: tribe.reset_to_tribal_population()
        if False:
            # clean these up or keep for documentation?
            self.ag_territories_striking_distance_NT = []
        # what should the s be for each tribe before collapse? the NC s? self.s? Maintain s at the moment

    def state_of_the_union(self):
        global whm
##s WHM.py:Confederation:state_of_the_union
        global whm
        if self.state is not POLITY_ALIVE:
            # constraint on state_of_the_union()
            raise RuntimeError("state_of_the_union() called for %s but not alive?!?" % self)
            
        for polity in self.member_polities:
            polity.state_of_the_union() # This is the one place where sou() gets called with POLITY_MEMBER
##s NADSM.py:PolityCommon:state_of_the_union
##s ComparativeHx.py:ComparablePolity:state_of_the_union
##s Population.py:DemographicPolity:state_of_the_union
##s WHM.py:Polity:state_of_the_union
        # only called when POLITY_ALIVE
        global whm
        self.end_year = whm.this_year # CRITICAL update to latest year alive
    

##e WHM.py:Polity:state_of_the_union
##< Population.py:DemographicPolity:state_of_the_union


        # CONSTRAINT udpate total_population here
        # NOTE the chronicler wants it for display during state_of_the_world()

        if self.size > 0: # member or alive: and self.state is POLITY_ALIVE:
            # Migration and total_pop is a per-polity activity
            
            # We migrate extant people in live polities and compute
            # total population here.  We grow people en-mass during state_of_the_world()

            self.update_arable_k()
            global whm
            if self.quasi_state:
                pass # we don't migrate quasi_states
            else:
                self.migrate_population(migration_factor=whm.migration_fraction)
            t_i = self.territories_i
            self.total_population = sum(whm.population[t_i])
            if self.total_population > self.max_population:
                self.max_population = self.total_population
            self.total_k = sum(whm.k[t_i])


    # This should be called whenever the set of territories changes
##e Population.py:DemographicPolity:state_of_the_union
##< ComparativeHx.py:ComparablePolity:state_of_the_union

        # Always collect the prediction data and reset below
        # define the criteria for counting this polity
        # We want it to be large enough but there is no restrictions on states being at least a century old
        # Thus we count it as a valid prediction even if there are several large_enough() states during the century
        if self.state == POLITY_ALIVE and self.large_enough(): # avoid counting any POLITY_MEMBERs
            global whm
            # TODO build a global index of territories in track_polity_biomes and update it whenever/if a territory changes biomes
            # then replace this calculation of regions with an np.intersect expression
            regions = []
            for territory in self.territories:
                if territory.Biome in whm.track_polity_biomes:
                    regions.extend(territory.regions)
            whm.predictions[regions] += 1
            global dump_predicted_history
            if dump_predicted_history:
                try:
                    int_this_year = int(whm.this_year)
                    century_i = whm.geography.years.index(int_this_year)
                except (ValueError, IndexError ):
                    pass
                else:
                    # BUG This records the id of the state at the century mark, not whether it has lasted a century
                    whm.predicted_history[regions,century_i] = self.id

# Mixing with WHM
##e ComparativeHx.py:ComparablePolity:state_of_the_union
##< NADSM.py:PolityCommon:state_of_the_union

        # state and population etc all updated
        # did state remain alive?
        if self.state is POLITY_ALIVE:
            global whm
            # record the maximum extent of this polity and the year it happened
            if self.size == self.max_size:
                self.max_territories_i = copy.copy(self.territories_i) # toplevel copy
                self.max_year = whm.this_year


    # maintain the extortion relationship between nomadic and agrarian polities
##e NADSM.py:PolityCommon:state_of_the_union
##< WHM.py:Confederation:state_of_the_union

        self.end_year = whm.this_year # CRITICAL update to latest year alive
    
##e WHM.py:Confederation:state_of_the_union
        if self.state is POLITY_ALIVE:
            # COLLAPSE: if there is a failed kurulthai, break apart and await another acceptable leader
            if whm.nomadic_succession_crisis_odds and whm.this_year >= self.succession_crisis_year:
                if urandom_random() <= whm.nomadic_succession_crisis_odds: # compute succession crises
                    self.collapse_to_tribes('lateral succession')
                    return
                else:
                    # this dynasty survived into another generation
                    self.succession_crisis_year = whm.this_year + whm.succession_generation_time

            # COLLAPSE: if not enough goodies from extortees to keep things together, collapse
            ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = self.ag_polities_within_striking_distance()
            if True:
                print('%s: %s extorts %s %s' % (whm.this_pretty_year,self,millions(total_extortable_population),ag_within_reach))
                
            total_extortable_population /= million
            if total_extortable_population < whm.confed_trigger_population:
                self.collapse_to_tribes('loss of agrarian wealth')
                return

            # MIGRATION: none
            # COMMAND AND CONTROL: compute the size of the confederation; distance no object
            self.max_territories_permitted = max(whm.confed_min_size,
                                                 min(whm.confed_max_size,
                                                     (total_extortable_population - whm.confed_trigger_population)*whm.confed_tribal_scale))

            # see if there are changes to whom we extort
            if whm.extort_only_max_pop_polity:
                if max_pop_polity not in self.extorting:
                    # remove the current current extortee and replace the with new guy
                    for extorting_polity in self.extorting: # no need to copy since it is a singelton
                        self.extort(extorting_polity,no_longer=True)
                    self.extort(max_pop_polity)
            else:
                # NOTE: slight variations in s adaptation rate by state might break asymmetry but nothing systematic ('cultural bias')
                for ag_polity in ag_within_reach:
                    if ag_polity not in self.extorting:
                        self.extort(ag_polity)

            # always go ahead of whoever (ag) matches this NC
            # increase effective s.  There will always be a max_s_polity as a confederation
            if max_s_polity.s >= self.s*whm.nomadic_s_threshold: # Are we threatened?
                self.s = whm.bounded_s(self,whm.nomadic_confed_s_increase,whm.max_polity_s)

            # Still alive...
            if False:
                # TODO compute the current K as a fraction of available_ag_p and distribute over all nomadic regions
                per_region_k_increase = available_ag_p*whm.nomadic_tax_rate/len(self.territories)
                whm.k[self.territories_i] = whm.nomadic_base_population + per_region_k_increase

    def machiavelli(self):
        return [] # NomadicConfederations don't War or JoinNomadicConfederation

class NomadicTribe(PolityCommon):
##f WHM:__repr__(...)
##f WHM:annex(...)
##f Population:compute_migration(...)
##f WHM:find_annexable(...)
##f WHM:find_borders(...)
##f WHM:large_enough(...)
##f WHM:make_quasi_state(...)
##f PopElites:migrate_population(...)
##f WHM:same_polity(...)
##f WHM:set_border_cache(...)
##f PopElites:update_arable_k(...)

    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
        # There are a fixed set of these asserted at the beginning of the run
        # As they are absorbed into confedations the tribes will oscillate between POLITY_ALIVE and POLITY_DEAD
## super calls PolityCommon:__init__(...)
        super().__init__(name=name,territories=territories)
        global whm
        self.name = 'NT %s' % self.name
        self.polity_type = POLITY_TYPE_NT
        self.large_enough_ignore = True # never count these toward statistics
        self.s = whm.min_polity_s # until powered up
        self.n_territories = len(territories) # this never changes
        self.tribal_connections = [] # cache
        self.ag_territories_striking_distance_NT = [] # territories within NT striking range
        self.ag_territories_striking_distance_NC = [] # territories within striking range if member of an NC
        self.has_horse_cavalry = False # mounted warfare has spread?
        self.first_confederation = None # DEBUG report when a tribe first joins a confederation
        self.reset_to_tribal_population()

    def reset_to_tribal_population(self):
        # NOTE we don't track population on the steppe...assume it is basically saturated for our purposes
        # Assume they are all elite
        global whm
        whm.elites[self.territories_i] = whm.nomadic_base_population
        whm.k[self.territories_i] = whm.nomadic_base_population
        whm.peasants[self.territories_i] = 0

    def collapse(self):
        raise RuntimeError("How can a nomadic tribe %s collapse??" % self)

    # @debug_on(AttributeError)
    def state_of_the_union(self):
        # COLLAPSE: tribes can never collapse to other tribes in this scheme

## super calls PolityCommon:state_of_the_union(...)
        super().state_of_the_union()
        global whm
        # if ALIVE (not MEMBER)
        if self.state is POLITY_ALIVE and self.has_horse_cavalry: # tribes can extort only when they get powered up
            # EXTORT:
            ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = self.ag_polities_within_striking_distance()
            self.total_extortable_population = total_extortable_population # save for confederation decision in machiavelli() below
            if whm.extort_only_max_pop_polity:
                if max_pop_polity is not None:
                    if max_pop_polity not in self.extorting:
                        for extorting_polity in self.extorting: # no need to copy since it is a singelton
                            self.extort(extorting_polity,no_longer=True)
                        self.extort(max_pop_polity)
            else:
                for ag_polity in ag_within_reach:
                    if ag_polity not in self.extorting:
                        self.extort(ag_polity)

            # always go ahead of whoever (ag) matches this tribe
            # increase effective s
            if max_s_polity is not None and max_s_polity.s >= self.s*whm.nomadic_s_threshold: # Are we threatened?
                self.s = whm.bounded_s(self,whm.nomadic_tribal_s_increase,whm.max_tribal_s)

            # CONFEDERATION: form if powered up and both wealth (people) and tribes are available
            if whm.nomadic_tribal_s_increase and self.total_extortable_population/million > whm.confed_trigger_population:
                del self.total_extortable_population # remove
                # See if we can form a minimum confederation
                connected_tribes = []
                frontier = [self]
                n_territories = self.n_territories
                while True:
                    if len(frontier) == 0 or n_territories >= whm.confed_min_size:
                        break # done
                    tribe = urandom_choice(list(frontier))
                    frontier.remove(tribe)
                    add_unique(connected_tribes,tribe)
                    n_territories += tribe.n_territories
                    for tc in tribe.tribal_connections:
                        if tc.confederation is None and not tc in connected_tribes:
                            add_unique(frontier,tc)

                if False and len(connected_tribes) > 0: # DEBUG so at least one more tribe than self
                    print('%s: NC test for %s c:%d t:%d %s from %s' % (whm.this_pretty_year,self,
                                                                       len(connected_tribes),n_territories,
                                                                       millions(total_extortable_population),ag_within_reach))

                if n_territories >= whm.confed_min_size:
                    # collect a set of tribes starting with self execute JoinNomadicConfederation against this 'empty' confedation
                    # that will maintain all the lists properly and recompute who they are extorting
                    confed = NomadicConfederation() # create the confederation
                    if True:
                        # open-code assert_core_polities
                        confed.core_polity = self
                        JoinNomadicConfederation(self,confed,'inner').execute()
                    else:
                        confed.assert_core_polities(core_polity=self);

                    # update the confederation's s with the max of the tribal s values
                    # then let the normal increment mechanism take over to move from there
                    for tribe in connected_tribes:
                        JoinNomadicConfederation(tribe,confed,'inner').execute()

                    # We have finally added 'territories' via NTs so update starting statistics
                    # Fragment from PolityCommon
                    confed.starting_size = len(confed.territories) 
                    confed.max_size = confed.starting_size
                    confed.max_territories_i = copy.copy(confed.territories_i) # initial value

                    # extort whoever is within reach
                    ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = confed.ag_polities_within_striking_distance()
                    if whm.extort_only_max_pop_polity:
                        confed.extort(max_pop_polity) # has to exist for the confederation to arise at all
                    else:
                        for ag_polity in ag_within_reach:
                            if ag_polity not in confed.extorting:
                                confed.extort(ag_polity)

                    confed.update_capital(confed.core_polity) # do this here
                    if True:
                        # Transfer the Clat/Clon from this triggering tribe to confederation rather than using the mean location of founding tribes
                        # Thus we can see the area the triggered it better
                        confed.Clon = self.Clon
                        confed.Clat = self.Clat
                    if True: # DEBUG
                        print('NC arises: %s %s @ %s,%s via %s sees %.1fM to extort (s=%.2f):' % (whm.this_pretty_year,confed,confed.Clon,confed.Clat,
                                                                                                  self,total_extortable_population/million,self.s))
                        for p in self.extorting:
                            print(' %s s=%.2f %d %.1f%% %.1fM' % (p,p.s,p.size,
                                                                  100.0*float(len(intersect(p.territories, self.ag_territories_striking_distance_NT)))/float(len(p.territories)),
                                                                  p.total_population/million))
                        if False: # DEBUG
                            print('%s composed of:' % confed)
                            for t in connected_tribes:
                                print(' %s' % t)
            
    def machiavelli(self):
        # ANNEXATION: there are no explicit battles with the extorted polities or territory expansion
        # just 'pressure' that is continuous and proportional to the nomadic state's population
        # and the population within striking distance of the extorted polities
        # So no orders for that.
        # However, if tribe is not a member of a neighboring confederation and the confederation has slots
        # it can opt to join as an order...
        orders = []
        if self.confederation is None:
            # JOIN:
            # Should this tribe join any on its border confederation?
            # This is NOT dependent on has_horse_cavalry since the confederation will teach you
            # collect all unique confederations whose # tribes is < their max
            # if any, select a random one and write an order to join
            # Actually it should be about *projected* tribal size after K increases when joining!!

            # This is the equivalent of NC.machiavelli() where the confederation looks at connections
            # from its member_polities to see if there are any tribes assuming the confederation
            # needs and can tolerate that tribe's number of territories.

            confederations = [] # which are hungry for more tribes?
            for confederation in unique([tr.confederation for tr in self.tribal_connections]):
                if (confederation is not None and
                    len(confederation.territories) + self.n_territories <= confederation.max_territories_permitted):
                    confederations.append(confederation)

            if len(confederations): # any nearby looking for a willing member?
                confederation = urandom_choice(confederations);
                orders.append(JoinNomadicConfederation(self,confederation,'outer'))
        else:
            # SECEDE: Tribes don't leave confederations
            pass

        return orders

# --- end of nomadic classes ----

class NADSM(CompareWithHistory,ElitePopulation):
##f WHM:advance_history(...)
##f PopChronicler:compute_world_population(...)
##f PopChronicler:compute_world_population_OLD(...)
##f ComparativeHx:finalize_display(...)
##f ComparativeHx:initialize_chronicler(...)
##f ComparativeHx:initialize_display(...)
##f WHM:load_geography(...)
##f Population:set_intensification(...)
##f Chronicler:set_pretty_years(...)
##f WHM:unfold(...)
##f PopChronicler:update_display(...)
##f ComparativeHx:update_display_for_trial(...)
##f Population:update_intensification(...)
##f PopElites:update_population(...)
##f WHM:verify_integrity(...)
    PolityType = AgrarianPolity
    TerritoryType = EliteOccupiedTerritory

    def __init__(self,options):
        options.description = 'Simulate nomadic and agrarian states in a geography'
        # options.add_argument('-m',metavar='model', help='Model name to use',default='NADSM')
        options.add_argument('--disable_nomads',action='store_true',help='Disable nomad formations',default=False)
        options.add_argument('--disable_nc',action='store_true',help='Disable nomad confederation formation',default=False)
##s WHM.py:WorldHistoricalModel:__init__
        self.start_time = time.time() # for simple elapsed time
        global whm
        whm = self # set package global
        # Move these to WHM so early set up and chronicler can see them
        options.add_argument('--show_figures',action='store_true',help='Display and save figures',default=False)
        options.add_argument('--make_movie',metavar='step',type=int ,help='Make a movie framed every step years',default=None) # typically 20
        options.add_argument('--last_year', metavar='year', type=int , help='Stop run in this year',default=None)
        options.add_argument('--compare_polity_size', metavar='#regions', type=int , help='Compare polities >= #regions',default=None)
        # Assume other classes have overridden argument declarations
        # TODO Add start_year, end_year, compare_size arguments here
        options = options.parse_args() # if you ask for --help parser calls sys.exit()

        self.options = options
        self.geography_name = options.geography_name
        X = [] # flatten the experiment list(s) to a single list
        for expr_lst in options.X:
            X.extend(expr_lst)
        options.X = X
        self.name = options.m # model name
        self.n_trials = options.n_trials
        
        # Start throat-clearing ...
        self.load_geography()
        # Do this after loading geography since it supplies data needed for parms
        self.initialize_parameters()
        print_scalar_members(self,header="Parameters after initialization:",footer="",indent="   ")

        self.establish_territories()
        self.initialize_chronicler()
        # End throat-clearing ...

        print('Setup time: %.1fs' % (time.time() - self.start_time))

        # Amortize the set up of above structures over several runs of unfold
        # which resets these instances.
        
        # setup for eval against other histories?
        # use deus_ex_machina() to assert initial polities
        # TODO is there any reason not to call self.unfold() immediately?
        # if no, change Utils.execute()

##e WHM.py:WorldHistoricalModel:__init__

    def establish_territories(self):
        # this runs after initialize_parameters so we can access contraints on connections
        geo = self.geography
        if geo.actual_lat_lon:
            lat = geo.lat
            lon = geo.lon
            if self.keep_sea_connections:
                # At this point sea_connections and desert_connections is a dict with region indicies as keys (and values)
                # remove them before they get converted to territories.  At this point we don't know any distances
                count_removed = 0
                for i in list(geo.sea_connections.keys()):
                    keep = False
                    for constraint in self.keep_sea_connections:
                        location,min_lon,max_lon,min_lat,max_lat = constraint
                        keep |= ((min_lon <= lon[i] <= max_lon) and (min_lat <= lat[i] <= max_lat))
                        if keep:
                            break
                    if not keep:
                        del geo.sea_connections[i]
                        count_removed += 1
                if count_removed:
                    locations = [c[0] for c in self.keep_sea_connections]
                    print('NOTE: Dropping %d sea connections not in the %s regions' % (count_removed,Oxford_comma(locations)))

            if self.keep_desert_connections:
                count_removed = 0
                for i in list(geo.desert_connections.keys()):
                    keep = False
                    for constraint in self.keep_desert_connections:
                        location,min_lon,max_lon,min_lat,max_lat = constraint
                        keep |= ((min_lon <= lon[i] <= max_lon) and (min_lat <= lat[i] <= max_lat))
                        if keep:
                            break
                    if not keep:
                        del geo.desert_connections[i]
                        count_removed += 1
                if count_removed:
                    locations = [c[0] for c in self.keep_desert_connections]
                    print('NOTE: Dropping %d desert connections not in the %s regions' % (count_removed,Oxford_comma(locations)))

##s Population.py:Population:establish_territories
##s WHM.py:WorldHistoricalModel:establish_territories
        # This version makes a 1:1 Territory for each geographical region
        geo = self.geography
        territories = [] # a list in geography region order; there could be duplicate Territories
        n_habitable = 0
        t_index = 0 # Set up indices in case there are global arrays
        for region in range(geo.n_regions):
            territory = self.TerritoryType([region])
            territory.t_index = t_index
            t_index += 1
            if territory.Biome == GEO_AGRICULTURAL:
                n_habitable += 1
            territories.append(territory)
        self.territories = territories
        # TODO map over land_connections and sea_connections
        # and create equivalent copied connection datastructures
        # now that all associated territories are created
        # ACTUAL: This is not really used by the actual model but we make it anyway
        for region in range(geo.n_regions):
            land_connections = []
            sea_connections = []
            desert_connections = []
            try:
                for neighbor in geo.land_connections[region]:
                    # create 'littoral' objects as needed
                    land_connections.append(Littoral(neighbor) if neighbor < 0 else territories[neighbor])
            except KeyError:
                pass # really?

            try:
                for entry in geo.sea_connections[region]:
                    sea_connections.append([territories[int(entry[0])],entry[1]])
            except KeyError:
                pass # ok, might not be littoral
            try:
                for entry in geo.desert_connections[region]:
                    desert_connections.append([territories[int(entry[0])],entry[1]])
            except KeyError:
                pass # ok, might not be littoral
            territories[region].land_connections   = land_connections
            territories[region].sea_connections    = sea_connections
            territories[region].desert_connections = desert_connections
        print('Established %d 1:1 territories, %d habitable' % (len(territories),n_habitable)) # DEBUG

##e WHM.py:WorldHistoricalModel:establish_territories
##< Population.py:Population:establish_territories

        # Now compute region lists for regional_k here
        # but use dxm to assert the values
        # This is a list used to generate population sums.  It is a subset of all the regions available
        self.region_list = ['Asia','India','Europe','Mesopotamia','Russia','sub_Saharan_Africa'] # for population reporting
        self.region_territories_i = {}
        self.region_territories_i['None'] = [] # NOP
        for region in geography_region_list: # WAS self.region_list: (see Geography.py)
            region_f = eval(region) # Get the function
            region_territories_i = []
            ag_region_territories_i = []
            for t in self.territories:
                if region_f(t.lat,t.lon):
                    region_territories_i.append(t.t_index)
                    if t.agricultural:
                        ag_region_territories_i.append(t.t_index)
            self.region_territories_i[region] = region_territories_i
            self.region_territories_i['Agr_%s' % region] = ag_region_territories_i

        region_territories_i = []
        for t in self.territories:
            # Wheat and rice don't do well in high elevations
            # (actually above 2km) but territories are still acquired there
            # We use this mechanism to avoid empries up in the high hills
            # This is largely tuned in China; the Alps are < 2km
            if t.agricultural and t.Elevation >= 3000:
                region_territories_i.append(t.t_index)
        self.region_territories_i['Agr_Mountains'] = region_territories_i

        # Turchin's PNAS incremental includes
        # NOTE: we assume 1:1 here and that the includes are the *same* as territory indices!
        geo = self.geography
        self.region_territories_i['Agr_expand1'] = setdiff(geo.PNAS_includes['Agr2'],geo.PNAS_includes['Agr1'])
        self.region_territories_i['Agr_expand2'] = setdiff(geo.PNAS_includes['Agr3'],geo.PNAS_includes['Agr2'])
    
##e Population.py:Population:establish_territories

        # at this point self has territories_i and territories and each territory has its lat/lon and connections
        # Needed for update_d() under PolityCommon:update_distance_from_capital()
        geo.sea_connections_i = np.array(list(geo.sea_connections.keys()))
        geo.desert_connections_i = np.array(list(geo.desert_connections.keys()))
        
    # compute  min distances from every non-steppe biome region to the nearest steppe region
    # we use those to compute striking distances, etc.  also, if the biome changes type ('desert' to ag)
    # we use it to update all nomadic tribe striking regions (and hence NC striking regions)
    # from which each NC computes its possible extortees
    def compute_min_steppe_distance(self,steppe_i):
        global whm
        geo = whm.geography
        non_steppe_i = setdiff(range(geo.n_regions),steppe_i) # order does not need to be canonical
        ns_lat = geo.lat[non_steppe_i]
        ns_lon = geo.lon[non_steppe_i]
        distance = np.ones(geo.n_regions,dtype=float )*1000 # 1k degrees (much greater than 360*sqrt(2) = 510 degrees)
        for s_i in steppe_i:
            d = np.sqrt((ns_lat - geo.lat[s_i])**2 + (ns_lon - geo.lon[s_i])**2)
            distance[non_steppe_i] = np.minimum(distance[non_steppe_i],d) # pairwise min
        distance[steppe_i] = 0
        return distance*self.km_distance_scale # km estimate

    def compute_tribes(self,tribal_i,tribal_sizes):
        # assigned contains either zero (unassignable to a tribe), -1 (not yet assigned) or the tribe number
        global whm
        geo = whm.geography
        assigned = np.zeros(geo.n_regions,dtype=int ) # UNAVAILABLE territory
        assigned[tribal_i] = AVAILABLE # not assigned yet but available for a tribe

        tribe_id = ASSIGNED # we replace this marker with the polity id after creation below for matlab use
        # There could be 'islands' of steppe/desert so make tribes in each of the islands
        while True:
            unassigned_i = [i for i in tribal_i if assigned[i] == AVAILABLE]
            if len(unassigned_i) == 0:
                break
            # Used to use set() and .add/.remove but that leads to non-convex polities
            # because order of addition and expansion is not preserved
            frontier = [] # the frontier along which new tribes can expand
            available = [] # the frontier along which a particular tribe can expand
            regions = []; # the set of regions for the current tribe

            # choose a starting location by first choosing a longitude and then either min or max lat within tribal_i
            s_i = urandom_choice(unassigned_i)
            s_lon = geo.lon[s_i]
            s_i = [i for i in unassigned_i if geo.lon[i] == s_lon]
            s_lat = np.min(geo.lat[s_i]) if urandom_random() > 0.5 else np.max(geo.lat[s_i])
            s_i = [i for i in s_i if geo.lat[i] == s_lat][0]
            # s_i should be a singleton point as the first seed

            tribal_size = urandom_choice(tribal_sizes)
            assigned[s_i] = tribe_id
            regions.append(s_i)
            for lc in geo.land_connections[s_i]: # use raw data, not Territory/Littoral
                if lc < 0 or assigned[lc] != AVAILABLE:
                    continue
                add_unique(frontier,lc)
                add_unique(available,lc)

            # add regions to a tribe randomly from the available list
            # if that is exhausted or the tribe is big enough, select
            # a random element from the complete frontier to expand
            while True:
                if len(available) == 0 or len(regions) >= tribal_size:
                    # close off the current tribe possibly start a new tribe
                    assigned[regions] = tribe_id
                    tribe_id += 1

                    if len(frontier) == 0:
                        break # done...
                    # Start a new tribe somewhere on the frontier
                    tribal_size = urandom_choice(tribal_sizes)
                    s_i = urandom_choice(list(frontier))
                    regions = [] # reset
                    available = [s_i]
                # expand the current tribe along its available frontier, which we know is unassigned
                # BUG: since we prematurely stop building and randomly extend we can get not very convex shapes
                s_i = available[0] # expand in addition order
                assigned[s_i] = tribe_id
                regions.append(s_i)
                available.remove(s_i)
                frontier.remove(s_i)
                lcs = geo.land_connections[s_i] # use raw data, not Territory/Littoral
                lcs = urandom_sample(lcs,len(lcs)) # permute the list of connections for stochastic construction
                for lc in lcs:
                    if lc < 0 or assigned[lc] != AVAILABLE:
                        continue
                    add_unique(frontier,lc)
                    add_unique(available,lc)
        return assigned
        

    def initialize_parameters(self):
        global whm,geo
        whm = self # set package globals
        geo = self.geography
        # set these options first
        self.disable_nomads = self.options.disable_nomads
        self.disable_nc_formation = self.options.disable_nc

##s ComparativeHx.py:CompareWithHistory:initialize_parameters
        global whm
        whm = self # set package global
##s PopElites.py:ElitePopulation:initialize_parameters
        global whm
        whm = self # set package global
##s Population.py:Population:initialize_parameters
        global whm
        whm = self # set package global
##s WHM.py:WorldHistoricalModel:initialize_parameters
        geo = self.geography
        years = getattr(geo,'years',None) # Assumes geography has this data
        # [Model]
        self.random_seed = None
        
        # [Time]
        if years is None:
            self.time_step = 2 # PNAS default
            self.first_year = -1500
            self.last_year  =  1500
        else: # geography supplies
            self.first_year = years[0]
            self.last_year  = years[-1]
            self.time_step = years[1] - years[0] # assume first step is uniform throughout

        if self.options.last_year is not None:
            # shorten total run time for debugging speed by limiting years
            self.last_year = self.options.last_year # debugging

        # For display and comparative purposes set the size of polity we care about
        if getattr(geo,'history',None) is not None:
            self.compare_polity_size = geo.min_polity_size # match historical data
            self.compare_polity_duration = self.time_step
        else:
            self.compare_polity_size = 2 # show polity only if as large as this (true polities)
            self.compare_polity_duration = self.time_step

        if self.options.compare_polity_size:
            self.compare_polity_size = self.options.compare_polity_size
            
        self.track_polity_biomes = all_biomes
        self.report_resurrected_polities = True
        self.tracking_biomes = biome_types(self.track_polity_biomes) # so saved data know...
##s Chronicler.py:Chronicler:initialize_parameters
        # [Display]
        self.display_show_figures = self.options.show_figures
        self.display_composite_figure = 2; # 0 individual figures, 1 - 1x2 with Area and Pop, 2 - 2x2 with Area/Pop Polities/R2
        self.display_time_interval = 20 # every 20 years
        # for movies and saving figures
        self.display_save_figures = self.display_show_figures

        # Setup various display parameters here for chronicler so recorded below
        if self.options.make_movie:
            self.display_show_figures = True
            self.display_make_movie = True
            self.display_time_interval = self.options.make_movie # must be supplied
            self.display_save_figures = True
        else:
            self.display_make_movie = False

        # TODO what are the values for a 30s movie (100 image per sec)
        # We have a frame image every display_time_interval years
        # An fps of 2 implies 2*frames per display second or
        # 40 years per display second for 20 year time interval
        # Thus 100 years per display second implies (100/display_time_interval)
        # years per display second or '25'
        # String to allow for 1/10, etc.
        # Since typically we run simulations over 3k years with a 20y frame window (150 frames)
        # here are the run lengths of movies at different fps
        # 1/2  300s (5.0m)
        #  1 - 150s (2.5m)
        #  2 -  75s (1.25m)
        #  3 -  50s
        #  4 -  38s
        # 12 -  13s # slides
        # 25 -   6s
        self.display_movie_fps = '3'
        self.display_resolution = 100 # DPI
        self.data_subdir = None # Used to form the subdirectory if needed
        # This serves as a tag for all the trials and should NOT be updated for each trial
        # We use UTC here as tags for figures
        self.date_tag = time.strftime("%m/%d/%Y %H:%M:%S UTC",time.gmtime(time.time()))
        print("Date: %s" % self.date_tag) # match the printed output to the graphs


##e Chronicler.py:Chronicler:initialize_parameters
##< WHM.py:WorldHistoricalModel:initialize_parameters


##e WHM.py:WorldHistoricalModel:initialize_parameters
##< Population.py:Population:initialize_parameters

        # CRITICAL that the time step be 'small' so population can edge up on k
        # else it will overshoot or not move at all for Historical
        # order matters here since we scale birth rate by the time step
        self.time_step = 2 # force this here until initialize_parameters(),setup_for_trial() logic is fixed

        # Set parameters according to anachronistic analysis of Medieval England 1150-1300CE:
        # At the start there were 1e6 people supported by 4 productive regions (4Mha)
        # At the end there were 3e6 people supported by those same 4 productive regions (Campbell, 2010)
        # To model K logistic growth properly we need to expand that by ~1.2 (or 3.6e6 potential)
        # These 4 regions were supporting people spread over 16-20 regions so we have density factor of ~4
        # (although England-Wales has ~30 regions total, not all were populated, especially in the north)

        # 11/7/14: Heard a lecture on modeling the Little Ice Age (LIA) and reports of a major volcanic eruption
        # in Indonesia in 1257 ('the year without summer') larger than Krakatoa that could have been the trigger
        # for local stress (and the Great Famine 1312 but that was 50yrs later?).  If K dropped after the eruption
        # then perhaps ME could have supported more people than 3M.  But it couldn't have been much more
        # 4M yields an agrarian population of 513M vs. the est of 395M using 3M (nearly exact w Kaplan). So below 4M...
        # but we still need to track people loss from warfare

        # so:
        # compute the K in people per km^2 based on ME data
        geo = self.geography
        self.unoccupied_density = 1/geo.equatorial_region_km2 # 1 person per region; not zero but not much

        # Great Britain is about 23Mha; England proper is 13.2Mha.  In medieval
        # times 5Mha of that (a few counties north and west of London) was (is)
        # the breadbasket and more or less fed the country.  We use population
        # estimates from Campbell to estimate a high point 'state' k. This
        # estimate is anachronistically applied throughout of agrarian OldWorld...

        km2_per_Mha = 1e5 # number of km2 per Mha
        # DEAD england_Mha = 13.2 # supported by 4Mha (perhaps 5Mha)
        england_Mha = 5.1 # medieval England supported by 5.1Mha in 1290 (12.75Ma, Broadberry and Campbell, 2011)
        england_km2 = england_Mha*km2_per_Mha

        # 10 regions occupied by the Normans between 1066 and 1300
        # with old constant 111.32^2 per region scaling, we estimated 10 'regions'
        # but the PNAS regions are PNAS_LATITUDE_SCALE*PNAS_LONGITUDE_SCALE rather than 1:1
        # and equatorial_region_km2 is 20Kkm2 rather than 12.39Kkm2 so we need only 12.39/20.0*10 = 6.1 regions
        if geo.PNAS_assumptions:
            # Fixed region size regardless of latitude.  Is this right?
            england_km2 =  9.3*(PNAS_LATITUDE_SCALE*PNAS_LONGITUDE_SCALE)*geo.equatorial_region_km2 
        else:
            # Regions covered by William's little northern adventure...
            # CRITICAL -- depends on the value of geo.equatorial_region_km2, which depends on actual_lat_lon and the version you emitted by create_geography
            england_regions = 10 # OLD
            england_regions = 9.3 # NEW -- bigger is lower density (might be better w/ 9)
            england_regions = 12 # Since we are not scaling in create_geography by LAT/LON scale we estimate we need an initial hinterland density of 4.
            england_km2 =  england_regions*np.cos(np.radians(52))*geo.equatorial_region_km2 # NEW -- increase this to decrease density
            print("Assuming England has %.1f regions or %skm^2" % (england_regions,millions(england_km2)))

        # Campbell, 2012
        src = "Campbell, 2012"
        population_1150 = 1e6
        population_1300 = 3e6 # 3.6
        # birth rate implied: 0.0175


        # Assume various inefficiencies (such as an elite 'tax') cause some percentage of K to be 'wasted'
        # ala Brenner.  We assume in this period the elites spend/waste 20% of K on luxury rather than their assigned social function
        max_k_scale = 1 + 0.2

        # Nice idea to line up with PNAS expand2 assumption but this is clearly too early for Asia, etc.
        # DEAD self.intensification_year = 900
        self.intensification_year = 1000 # 1000CE 
        self.intensification_improvement = 2.0 # doubling yields

        # BUG This is incorrect....see create_geography()
        # In fact, this is the k of an state of 1 region, not hinterland, which doesn't have to putative cultural improvments
        # true hinterland is probably 1.43 vs. 4.3
        # for an estimate based on world population and arable land at the time
        # assuming a low uniform density
        self.hinterland_density = (max_k_scale*population_1150/england_km2)
        # JSB's quasi-uniform K hypothesis, see OccupiedTerritory init() for the quasi bit
        # We are estimating K from population; Turchin points out that they never reach it to within 20%
        self.state_density     = (max_k_scale*population_1300/england_km2)
        if self.intensification_year < 1150: 
            self.hinterland_density /= self.intensification_improvement
            self.state_density      /= self.intensification_improvement


        self.hinterland_density_factor    = {} # year -> factor

        # An interesting experiment: Assume the k even for agricultural hinterland
        # is the same as state but keep the hinterland birthrate low (1/25)
        # In this case the world population rises nearly linearly to 311/160M under HistoricalPop
        # this appears not so good, especially before 0CE (high) 
        # But remember the model that fluctuates between high state and lower hinterland
        # causes death as states collapse and here we defeat that death--we just slow down life
        # self.hinterland_density = self.state_density # EXPERIMENT

        # This is apparent starting density to match Kaplan's estimates for 1500BCE
        # given our agrarian tags over the OldWorld (dominated mostly by hinterland)
        self.initial_population_density = 1.45 # people/km2 (OldWorld create_geography stats)

        # Now as a check under PNAS include assumptions:
        # We have 1852 initial AG regions in 1500BCE nearly all in hinterland
        # and we assumed 62.5K people per region yielding an initial pop of 115.75M
        # which comports well with Kaplan, 2014 but overstates Kremer, 1991 by 4x: 27M
        # Finally, in 1500CE, PNAS has 2500 AG regions almost all under state
        # we estimated 225K per state region hence 562.50M possible world population
        # which comports well again with Kaplan and Kremer at around 500M

        # Under population density and area_km2 assumptions:
        # HistoricalPop w/ OldWorld shows total agrarian land area is ~43740254km2 (43.7Mkm2)
        # Using hinterland K of 1.45/km2 you get 63M which is of course from Kaplan since that is what we used to get it
        # If all eventually grew to England start level of 3.33 this would be 144M at max
        # Under state K (late England level) of 8.17 is 357M which is good compared with Kaplan (386M) but low for Kremer 500M
        # However certain regions (China/India/Europe) increased their population after 1000CE and in spite of the Black Plague and Great Famine
        # create_geography estimates density of 2.95/km2 in 1500BCE and 17.9/km2 in 1500CE

        # The current code assumes the following:
        # Hinterland can sustain 3.3 but starts at 1.45 with a very low birthrate
        # If an state falls back to 3.3 (from 8.17) the birthrate also falls.  So really we assume there is a natural K
        # from hinterland agcriculture and other foraging that is amplified under the state but the birth rate
        # under the two state conditions is very different on average, possibly due to stability of state in spite of 'coercion'

        # From fitting the logistic curve over 150 years (1150 to 1300CE) between 1e6 and 3e6 people
        self.base_birth_rate = logistic_beta(population_1150,population_1300,population_1300*max_k_scale,1300-1150) # under Campbell 0.0175  1.75% net birth rate per year
        print("Using %s estimates of medieval England population (%s at peak) implies birth rate of %.2f%%/yr" % (src,millions(population_1300),self.base_birth_rate*100.0))

        # CRITICAL Analysis:
        #  You must have a small starting population in hinterland to match initial demographic data
        #  Hinterland must grow much slower, basically not at all, in order to keep the pop growth down
        # How realistic is this?  The implicit conjecture is that when you become an state of 2 or more regions
        # that that organizational scheme, whatever it is, permits the birth rate to increase by a factor of 25...
        # These values were gathered using HistoricalPop and varying fraction
        # /100 yields 254/104M
        # / 50 yields 257/108M and ~ same curve as /100
        # / 25 yields 260/112M and ~ same curve as /100
        # -- this is about the break point
        # / 10 yields 256/109M and elevated version of curve that flattens at the end
        # /  1 yields 231M/89M a large initial explosion not seen and max population is
        # See Kaplan/hinterland_birth_rate.m which estimates rate at 1.71%/8.55 (we use 10)
        self.hinterland_birth_rate_factor = 10

        # Specify a migration rate PER TIME STEP
        # This is not like birth rate which scales with time step
        # Instead this is the maximum number of extant people that can be shuffled about in a time step
        # That can't exceed 100% (migration doesn't create or destroy any people)
        # Mostly stationary since until high population pressures elites wanted to keep/coerce the population
        # to work their land.  But when there was new frontier opportunities some elite moved with some people
        self.migration_fraction = .002 # .2% [people per state per time step]

        # In the case where k collapses (state collapse typically but also famines)
        # should the population crash as well?
        # If parameter is 0, the massive die-off logic is disabled and they die off slowly
        # at (inverse, logistic) birthrate associated with new polity state; otherwise immediate
        # massive die off to this fraction of new k.  Should be less than 1 but not too low!
        # This would typically be less than or equal to a misery amount
        self.k_collapse_fraction = .80

        # regions only apply to actual worlds from create_geography, not create_arena
        # ok this is complicated: normally if 1 we just set the initial base intensifications in the first year
        # this needs its own base scale factor
        # then there is when the expand1 and expand2 regions increase their base productivities
        # and when there is the world-wide doubling (which in the base coincided with expand2)
        # TODO so we need to tease these things apart
        # Right now: expand1 always happens at 300 and expand2 at intensification_year
        # and the 2x skips! those regions
        self.use_regional_k = 2 if self.geography.actual_history else 0 # we also permit addition values like 1, etc.
        self.regional_intensifications = {}

        # TODO if actual_history do we (1) spread total_population over starting world? (2) do we create starting states
        # HistoricalPop: yes, yes?!
        # ADSM: yes, yes
        # PNAS: yes, no (just hinterland) (Thus PNAS overrides this)
        # A2: no, no (not actual history so skipped)
        self.create_initial_states_from_history = self.geography.actual_history

        # How to scale the initial starting population
        self.initial_population_scale = 1.0

##e Population.py:Population:initialize_parameters
##< PopElites.py:ElitePopulation:initialize_parameters


        # CRITICAL that the time step be 'small' so population can edge up on k
        # else it will overshoot or not move at all for Historical
        # Also order matters here since we scale birth rate by the time step
        self.time_step = 2 # follow PNAS lead

        # Elite fraction of K typical for agrarian states
        # This is an average and reflects the typical size of the support
        # entourage required to support the protective/extractive function of
        # the actual (smaller number of) elites
        # This reflects the 'non-producive' labor that is somehow supported by labor/taxes on peasant surplus
        # This fraction indirectly controls the power projected on borders by elite population for the ADSM model
        self.base_elite_k_fraction = .2 # 20% using the 80/20 rule
        # This can be dropped to 1% without impact
        # As long as everyone scales their elites by the same fraction this just scales the armies
        # self.base_elite_k_fraction = 0.05 # Koyama's review of Scheidel's Escape from Rome estimated 3-4% of population to the warrior class

        # implement separate migration rates for peasants and elites
        # but make them the same here as in the base model
        self.migration_fraction_elites   = self.migration_fraction
        self.migration_fraction_peasants = self.migration_fraction


##e PopElites.py:ElitePopulation:initialize_parameters
##< ComparativeHx.py:CompareWithHistory:initialize_parameters


##e ComparativeHx.py:CompareWithHistory:initialize_parameters

        # If making a movie, mark polity id in movie at Clat,Clon
        # None disables, [] shows all, else a list of specific polity ids
        self.display_marked_polities = []
        self.compare_polity_duration = 75 # Must have lasted for 3 generations already at a century mark to be counted

        # computational variants

        # Alternative experiment to avoid_hinterland_hinterland_annexation:
        # In the case of Hinterland getting more powerful than its neighbors
        # what are the odds of a new state arising from Hinterland?
        self.hinterland_arise = 1.0 # DEFAULT 1.0, which means a state is always created
        # self.hinterland_arise = 0.0 # no chance which means only states via collapse can be created
        # self.hinterland_arise = 0.2 # 20% chance

        # When specifying constants here and in setup_for_trial methods assume this scale
        self.km_distance_scale = 100

        # Override values from Population.py
        # KK10 appears to overstate population in 1500BCE (via interpolation) so scale back
        self.initial_population_scale = 0.4
        # This is critical to get the number of early states down
        # The base birthrate from Medieval England (1.7%) appears excessive for states in earlier periods
        # but they don't have the power to acquire more area (s=1.1) so they acquire hinterland slowly
        # this means they collapse early and because there is no place for population to migrate then collapse often
        self.base_birth_rate = 0.0135
        # Normally the 0.2%/time_step is good for elites (else you have an impact on DST)
        # but we could move peasants more frequently to deal with population issues and keep from the malthusian limit
        # BUT that might get reflected in the total overall population
        # Actually, it doesn't really, even at 100 times the rate
        self.migration_fraction_elites   = self.migration_fraction
        self.migration_fraction_peasants = self.migration_fraction*100 # more likely to migrate

        self.create_initial_states_from_history = True # Gotta have a start
        # year -> a list of instructions to initialize states
        # This covers both the create_geography and the create_arena examples
        # Population.py does the automatic creation based on history unless this is overridden (see ADSMX)
        # DEBUG self.create_states_instructions = {} # None -> initialize from historical states
        initial_state_sizes = np.nan
        self.create_states_instructions =   {-1500: [[('EgyptNew', -1500), initial_state_sizes, self.PolityType], # Nile Valley large extent
                                                     [('Neo-Assyria', np.nan), initial_state_sizes, self.PolityType], # Tigris/Ephrates
                                                     [('Hittite', -1500), initial_state_sizes, self.PolityType],],
                                             -1200: [[('Shang', -1500), initial_state_sizes, self.PolityType],], # MYRV
                                              }
        self.succession_generation_time = generation_years # number of years a typical king/shangyu rules before a successor/kurulthai is called

        # NOTE the track_polity_biomes need to be set here, NOT per trial
        # because we call initialize_chronicler() right after this in WHM and it needs this set
        # NADSMX sets this in its initialize_parameters() after this is called as supers()
        # So look there for the proper settings!!
        self.track_polity_biomes = agst_biomes # track agrarian and nomadic states

        # The main innovation in the NADSM model are the replacement of prior, exogenous s increases
        # with an 'autocatalytic' model of s increase triggered by nomads inventing horse cavalry
        # and extorting nearby agrarian states in the two main spreading centers at the proper times and strengths
        # See all the parameters that control the s calculus below, after main agrarian and nomadic polity parameters

        # Parameters controlling agrarian polity growth: war and people
        # War:

        # Rather than a stochastic approach to victory we take a more conservative approach
        # there might be many skirmishs but true victory occurs when material advantages dominate and the victor can hold it
        self.War_victory = 1.0 + 0.2 # power must be 20% more than the defender to win

        self.War_min_intensification = 0.6 # estimated yield worth fighting for...
        self.War_max_elevation = 3000; # (m) avoid war of above this elevation (effectively disabled)

        # What fraction of the total number of border regions to attack per time step?
        # TREND: As fraction decreases from 1/2 to 1/3 the number of small states that last a century decreases
        self.attack_budget_fraction = 0.45

        # per exchange, what fraction of the projected army is wasted
        # the deaths are distributed uniformly throughout the remaining regions of the vanquished, if any, before polity swap
        # the consequence is that as attacks proceed closer to the vanquished depot the more people are wasted and
        # the country as a whole depletes approximately exponentially
        # NOTE the cost/benefit of successful war under parameters is 8:1
        self.vanquished_death_fraction = 0.8
        self.victorious_death_fraction = 0.1

        # All sea and desert battles start off disabled (no distance and no shortening)
        # But you can set ta schedule and/or increment
        
        # All sea battles in the Mediterranean and desert battles in the Middle East
        # See ADSM.load_geography() for code that restricts desert and sea connections to those lat/lon regions (happens before we get here)
        # You can still disable sea and/or desert battles by setting these parameters to zero
        # distance factors scales how much shorter a connection appears because of mode of transport:
        # 1.0 means no shrinkage, 0.5 means half the distance, etc. 0 disables the particular calculation (equivalent to 1.0 but without the computation)
        # NOTE: sea battles were prevalent in this period in the Med so no timing restriction
        # The distance from Carthage to Rome, about 670km
        # Alternatively to get from 140km to 670km by 0CE you should increment (/ (- 670 140.0) (/ 1500 2.0)) == 0.71 km/time_step
        # ADSM stock, last investigation against Mediterranean
        self.keep_sea_connections = [('Mediterranean', -10.0, 40.0, 28.0, 47.0)]
        self.sea_distance_schedule = {-1500: (1*geo.interregion_distance_km, .3),
                                      -1000: (2*geo.interregion_distance_km, .3),
                                      -500: (3*geo.interregion_distance_km, .3)}
        self.sea_battle_distance_increment = 0 # in km/time_step

        self.keep_desert_connections = [('Levant',35.0,75.0,23.0,45.0),('Sogdiana',60.0,90.0,38.0,48.0)] 
        # Harl notes that the camel was not domesticated and hence could not support desert warfare until 900BCE
        self.desert_distance_schedule = {-1500: (200,0.5),
                                         -900:  (400,0.3)} 
        self.desert_battle_distance_increment = 0 # in km/time_step

        # Parameters controlling agrarian polity collapse
        self.base_misery_threshold = 0.85 # What fraction of opportunity saturation initiates miserable elites
        self.faction_increased_asabiya_time = generation_years # number of years a new faction is pumped up after formation
        self.faction_death_vs_tolerance = 0.8 # fraction of increased time gained by death vs increaed misery tolerance
        
        # self.faction_odds = [2, 2, 2, 2, 3, 3, 3, 4, 4, 5] # odds: .4 for 2, .3 for 3 .2 for 4 .1 for 5
        self.faction_odds = [2]
        # how likely is it for an unassigned region to join a particular faction?
        # Note that the paper talks about the chance to pass adding a region, so it is describing 1 - faction_add_odds
        # For 50% it doesn't matter.
        self.faction_add_odds = 1.0 # always (so equal sized factions rougly)
        # self.faction_add_odds = 0.9 # 10% chance of a miss, generating somewhat unequal sized) factions
        # self.faction_add_odds = 0.5 # 50% chance of a miss, generating somewhat unequal sized) factions

        # This may scale with s!?
        # self.faction_collapse_odds = 0.00 # chance that a faction collapses to hinterland after formation
        self.faction_collapse_odds = 0.25 # from TCTG13 investigation

        # Parameters that support nomadic tribes and confederations
        # Parameters to control creation of tribes
        # These are empirical estimates from the gross statistics in the polity record
        self.OldWorld_number_of_tribal_regions = OldWorld_number_of_tribal_regions # record assumptions
        self.OldWorld_number_of_tribes = OldWorld_number_of_tribes # record assumptions
        self.tribal_distance_from_steppe = 500 # how close can tribes live from steppe (into desert) km from GEO_STEPPES (used to compose initial tribes)
        self.min_tribal_size = OldWorld_number_of_tribal_regions/OldWorld_number_of_tribes # regions (31 regions per tribe or about 600Kkm2)
        self.max_tribal_size = 1.6*self.min_tribal_size # regions (49.6 per tribe 1Mkm2)

        self.confed_min_size = 5*self.min_tribal_size # about 5 'inner' tribes regions (155 regions)
        # from macrostate data, no steppe nomadic state is larger than this regions (311 regions)
        # This implies at most an additional 5 'outer' tribes (311 - 155)/30
        # self.confed_max_size = self.OldWorld_number_of_tribal_regions/3 # roughly 3 very large NCs at maximum saturation
        # running nc_historical_scale.m we find the max size NC occupied 49% of the entire steppe; most were smaller
        self.confed_max_size = self.OldWorld_number_of_tribal_regions/2 # 

        self.extort_only_max_pop_polity = False # if False extort all polities within striking distance
        self.tribal_strike_distance = 200 # how far can individual nomadic tribes strike from their territories into agrarian polities (km)
        # zero scale requires an explicit confed_strike_distance 
        self.confed_strike_distance_scale = 0 # UNUSED What fraction (typically > 1, e.g., 2.3) of the tribal strike distance can confederations strike
        # Between 400 and 600km (steppe to Babylon)
        self.confed_strike_distance = 550 # how far can nomadic confederations strike from their territories into agrarian polities (km)

        # when do nomads develop horse cavalry and how long to spread to the other tribes? (It spreads into ag stats via extortion s increase)
        # From Turchin, et al. 2016 'The Spread of Mounted Warfare' JClio
        # Mounted horse archers and tactics in 1000BCE in PC
        # See cav_spread_lon.png cav_spread_lat.png

        # The emergent dynamics are: NT invent horse cavalry, small states get pestered at a distance, they expand away and toward NT faster (^s)
        # while more NT get archers (diffusion), eventually pop clears 5M locally and boom! NC arises
        # the trigger population threshold must vary to account for the earlier start time so things raise in 600BCE
        cavalry_start_year = -1000 # year when first steppe nomad territory perfects mounted warfare
        cavarly_china_year =  -400 # Year when mounted warfare arrives on NW China border (see Turchin data)
        # cavarly_china_year = -200 # A test to see how the other statistics change of NC in the east is delayed
        # (W lon, E lon, year, extortable pop (M))
        self.mounted_warfare_invention = (Pontic_Caspian_lon,Pontic_Caspian_lon,cavalry_start_year,None) # stock (start spreading in PC in 1000BCE)
        # DEAD self.mounted_warfare_invention_longitude = Pontic_Caspian_lon
        # DEAD self.mounted_warfare_diffusion_start_year = cavalry_start_year
        self.mounted_warfare_diffusion_time = (cavarly_china_year - cavalry_start_year)

        # reset these working variables in case of MP processing
        self.eastern_cavalry_diffusion_limit = None
        self.western_cavalry_diffusion_limit = None
        self.mounted_warfare_diffusion_end_year = None

        # There is some evidence that this number grew to ~3-4M around 1500CE but it isn't clear if that is because
        # of trade support or actual productivity increases.  probably trade.  In any case, compared with 450M in all of Eurasia
        # we can neglect this growth.  Especially since our nomadic military influence scheme does not track
        # military deaths (considered negligible) or or scales with people (rather by tribe)
        self.nomadic_base_population = million/OldWorld_number_of_tribal_regions # estimated population over steppe/desert regions in 1500BCE
        # Assuming the trigger is 15M for the min NC and the max NC size is reached with 25M of extorted population
        # we have a size rate of about 15 regions per 1M of extorted population or about 1 tribe per 2M
        # 5M seems to trigger Persia at the right time after some period of NT abuse
        self.confed_trigger_population = 6.3 # total population (millions) within tribal reach required to raise a confederation
        self.confed_max_population = 200 # (millions) control the *rate* which a confederation can acquire tribes toward the confed_max_size (in regions)
        self.fraction_ag_state_terriorties_within_strike_distance = .1 # 10% of a state exposed before exploitation pressure is felt
        self.nomadic_succession_crisis_odds = 1.0/6.0 # odds that a new tribal 'dynasty' will supplant the current NC and cause its collapse (6 generations)

        # Parameters that control the increase in s because of nomadic military shock ala Turchin 2009
        # Here is the basic scheme:
        # Initially all NTs and initial agraian states start out at some initial_ag_state_s value
        # In mounted_warfare_diffusion_start_year horse cavalry is invented on the steppe so some NTs in the Pontic Caspian (or wherever) region
        # get powered up.  This means they can increase their apparent s as described below up to some max_tribal_s and pester/extort nearby
        # agrarian states, if any, that have at least 10% of their land withing the NT striking distance (200km typically).
        # That apparent s increase causes the extorted agrarian states, if any, to increase their own s at some rate governed by
        # the s difference and a rate parameters s_adapt_factor.  The nomads see the increasing s and when the agrarian states get 'close'
        # they increase their s by a small increment, which causes the agrarian states again to follow.  Thus in this model, the nomads
        # are the initial and continuing cause of increasing s.  When an NT sees the extorted population grow beyond a size threshold
        # it confederates with other tribes, which increases the combined striking distance (to maintain its extortion footprint of ~10%)
        # and raises the maximum s they can reach to so max_polity_s.  In addition to this basic scheme, all agrarian states look at the
        # s values of their political neighbors and if there is a positive discrepancy (nomadic or agrarian) they begin to adapt.  This 'knock-on'
        # effect spreads the implementation of higher military efficiency across the agrarian world from the initial nomadic shock
        # at a rate that depends on the adaption factor but actually more on the spread of large scale states, hence via territory annexation.
        # When nomadic tribes confederate a simple thing happens: all tribes rise to the level of the maximum tribal s of the confederation
        # However, there is no 'knock-on' between the nomadic confederations.


        initial_ag_state_s = 1.1 # better slow start
        max_tribal_s = initial_ag_state_s*1.3 # should be less than largest s (30% larger)
        # Hegemonic bounds in SM suggest 2.65, which would generate Yuan-sized states which is too big
        # (but must be biggger than 1.95, which is ADSM)
        maximum_ag_state_s = 2.1
        self.min_polity_s = initial_ag_state_s # for initial agrarian and nomadic tribe polities
        self.max_polity_s = maximum_ag_state_s # for all polity types; implied historical maximum by Yuan

        # How to scale state-level s for hinterland (wrt to min_polity_s)
        self.hinterland_s_scale_factor = .5

        # This is bounded by 0 (never change, i.e., ADSM) and 1 (immediately match); greater than 1 implies an immediate overshoot
        # estimated time to double s
        # 0.005 600+ yrs
        # 0.010 400 yrs 16G
        # 0.050 150 yrs 6G
        # 0.100  75 yrs 3G 
        # 0.005  50 yrs 2G
        self.s_adapt_factor = .10 # how fast agrarian states adapt to s threats per time step (fraction of s)

        # Another potential source of s increase is 'endogenously' via the Kremer mechanism of inventions according to population
        # In our implementation we assume that knowledge of military inventions/innovations made anywhere in the world
        # spread immediately and are adapted by all extant states uniformly, a clearly poor model.  Mostly to demonstrate
        # there had to be 'motivated' invention due to nomadic shock (the military version of Boserup's principle).
        
        # Kremer's actual eqn is ds/dt = g*P*s, where s is the level of technology (here assumed to be military only)
        # (which yields increased production via adding additional land but at a constant level of production)
        # P is current population and g is some constant
        # See kremer_s.m to estimate g based on Old World population and starting at min_polity_s and max_polity_s
        # g = 1.53e-10; % military inventions per person per century (empirical fit to reach 2.65 in 1500CE)
        # self.time_step*(1.53e-10/100) # convert g from per century to per time step
        self.s_kremer = 0 # update s each time tick based on total_population

        # The nomadic side of Turchin's auto-catalytic idea
        # Each NT/NC looks at the max s of the states it extorts, which could be increasing (because of it or someone else)
        # if that s gets within 95% of the nomad's capability, it increases it step-wise by the various increase parameters below
        # up to max_polity_s
        # NOTE: The actual rate of increase over time is actually governed by s_adapt_factor on the agrarian side
        # To disable nomadic-leading increases set this to 1.0 or larger (and set the increases to 0 as well)
        self.nomadic_s_threshold = 0.95 # if an ag state s is within this percentage of NT/NC's current s, increase its s by the following increments
        # A value of 0.05 implies that to change an s from 1 to 2 it takes (2 - 1)/0.05 = 20 time steps hence 40 years *assuming*
        # that at each step the neighbor always adapts fast enough to trigger the nomadic_s_threshold of 0.95 each step (s_adapt_factor = 1.0)
        # otherwise it will take a longer time
        self.nomadic_tribal_s_increase = 0.03 # how much to increase effective s of an NT when an ag gets close
        self.nomadic_confed_s_increase = 0.05 # how much to increase effective s of a NC when an ag gets close
        # When an NT receives the gift of horse cavalry what is the maximum s it can achieve and threaten an agrarian state with?
        self.max_tribal_s = max_tribal_s # What is the new max s permitted?
        # end of nomadic confederation parameters

    # Ensure a polity's changed s doesn't exceed the min or specified maximum s (unless it already was)
    def bounded_s(self,polity,increment,max_s):
        ps = polity.s
        ns = ps + increment
        ns = min (max (ns, self.min_polity_s), max_s)
        return ps if ps > ns else ns
    
    def update_power_projection(self):
        this_year = self.this_year
        # HEGEMON setting self.s (aka whm.s) sets the default scale for states and hinterland
        # NOTE that this calculation is done whenever power projection is performed
        # so each polity does NOT latch its s at the start of its life but improves as the world improves
        if self.s_kremer != 0:
            # Kremer, 1993 Equation 3 suggested that inventions occurred at a rate proportional to the population
            # ds/dt ~ gsP
            # Here the constant converts inventions into increased s, which then spreads by knock-on
            # this sets the s for all new (agrarian) polities; older polities will learn from them so no broadcast needed
            self.s = self.bounded_s(self,self.s_kremer*np.sum(self.population)*self.s,whm.max_polity_s)

        # In the past we had exogenous date-based changes to s, which impacted Hinterland as well
        # In NADSM the initial s for Hinterland (min_polity_s) never changes so compute Hinterland power projection once
        if this_year > self.first_year:
            return # already done...

        # Compute the hinterland power scaling indexed by latitude
        self.power_scale_1 = {} # reset
        s1 = self.min_polity_s*self.hinterland_s_scale_factor
        self.Hinterland.s = s1;
        print('%s: Computing single-region distance in km, scaled by %dkm (effective s = %.3f)' % (self.this_pretty_year,self.km_distance_scale,s1))
        distance = self.geography.lat_km_per_degree/self.km_distance_scale
        # scale the single-region power by latitude
        for lat in self.unique_lats:
            self.power_scale_1[lat] = power_projection_factor(distance*np.cos(np.radians(lat)),s1)
        
    def setup_regions_for_trial(self):
        # Population has set up regional_intensifications to inform where we might have problems
        if self.use_regional_k:
            for year in np.sort(list(self.regional_intensifications.keys())):
                for region,factor in self.regional_intensifications[year]:
                    if factor < self.War_min_intensification:
                        print(" Avoiding annexation of %s regions in %s with intensification of %.2f" % (region,pretty_year(year),factor))

        # Execute after supers() because we need Hinterland and Unoccupied polities
        # In this simulation we use the Biome to distinguish the proper polity to assert
        # All territories start out unoccupied unless GEO_AGRICULTURAL
        # This is mainly for eventualy Population tracking we assert K in Unoccupied as zero, and Hinterland non-zero.
        for territory in self.territories:
            t_i = territory.t_index
            if territory.Biome == GEO_AGRICULTURAL:
                polity = self.Hinterland
            else:
                polity = self.Unoccupied
            territory.set_polity(polity)
            # setup lat/lon arrays for power projection calculation
            # these lat/lon is the mean of the underlying regions
            self.latitude[t_i]  = territory.lat
            self.longitude[t_i] = territory.lon

        # compute initial hinterland costs assuming no sea or desert battles possible
        hpo_t,hph_t = self.Hinterland.find_borders(sea=0,desert=0,physical=True)
        # VN assumption implies 4 connections per interior regions and assume 3 political connections for each physical (one ocean, desert, etc.)
        global hinterland_border_cost
        nVN = float(getattr(self.geography,'n_land_connections',4)) # assume original VN (NSEW) if not explicit
        hinterland_border_cost = (nVN*(self.Hinterland.size - len(hph_t)) + (nVN - 1)*len(hph_t))/self.Hinterland.size
        self.hinterland_border_cost = hinterland_border_cost
        self.states_created = set()

        # ADSM turns out with regional the 'seam' between Europe and Russia get populated with
        # Europe having 1.0 intensity and Russia with 0.3.  Even with the area_km2 identical this difference
        # sets up Europe to have more hinterland elites than Russia and Europe can invade.  And then
        # Europe rips through Russia to the north, destroying the statistics, etc.
        # But really there is nothing for Europe to gain....
        # We can avoid all this with smoother intensification transistions
        # and/or we can avoid_hinterland_hinterland_annexation (HACK) so single hinterland regions
        # can not rise as states but can only be annexed
        if self.use_regional_k:
            if avoid_hinterland_hinterland_annexation:
                print("WARNING: Avoiding hinterland/hinterland annexation!")
            else:
                # If we assume all agrarian hinterland gets assigned the *same* initial population
                # (regardless of intensification -- see set_polity() above) then we will have
                # spontaneous states only if there is growth since eventually on the seams
                # one hinterland regions will have enough people to dominate its neighbor with
                # less intensification. Report where those places might be and how long before it might show up:
                if self.hinterland_birth_rate > 0:
                    # In time_steps; all hinterland grow at the same rate, regardless of the absolute K they eventually reach
                    # the start ratio of hinterland to its eventual 1.0 k
                    P0 = self.initial_population_density/self.hinterland_density
                    simulation_time = self.last_year - self.first_year # elapsed years of expected simulation
                    intensification_values = []
                    for year,instructions in self.regional_intensifications.items():
                        for region,factor in instructions:
                            intensification_values.append(factor)
                    intensification_values.append(1.0); # default starting value
                    intensification_values = np.sort(unique(intensification_values)) # ascending order
                    n_values = len(intensification_values)
                    mean_power_scale_1 = np.mean(list(self.power_scale_1.values())) # over all latitudes
                    for i in range(n_values):
                        defensive_intensification = intensification_values[i]
                        if defensive_intensification < self.War_min_intensification:
                            continue # no one will care
                        for j in range(i+1,n_values):
                            offensive_intensification = intensification_values[j]
                            # When does a hinterland region with larger intensification become militarily dominant over a defensive neighbor?
                            # Just because of population density assuming the same s:
                            # offensive_intensification*power_scale(1)/defensive_intensification > self.War_victory
                            # 1.0 is K so 1.0 > (defensive_intensification*self.War_victory)/(offensive_intensification*power_scale(1))
                            scaled_power_ratio = (self.War_victory*defensive_intensification)/(mean_power_scale_1*offensive_intensification)
                            # solve logistic_t for the time it will take to exceed this scaled_power_ratio in years...
                            if scaled_power_ratio > 1.0:
                                continue # infinity....
                            time_to_dominate = logistic_t(P0,scaled_power_ratio,1.0,(self.hinterland_birth_rate/self.time_step))
                            # DEAD time_to_dominate = np.log((scaled_power_ratio*(1.0 - P0))/(P0*(1.0 - scaled_power_ratio)))/(self.hinterland_birth_rate/self.time_step)
                            if time_to_dominate < simulation_time:
                                # If we were clever we would report the border of territories (Europe over Russia)
                                # if they abutt but we don't know if they do (and won't until assignment in set_polity() above)
                                print("WARNING: Hinterland with %.2f intensification will dominate %.2f in %.1f years" % (offensive_intensification,
                                                                                                                          defensive_intensification,
                                                                                                                          max(0,time_to_dominate)))

    def setup_nomads_for_trial(self):
        # Create nomadic tribes
        if self.tribal_strike_distance > self.confed_strike_distance:
            raise RuntimeError("How can tribal strike distance be greater than a confederations distance?")

        self.nomadic_tribes = []
        # given typical parameters above: (467 - 155)/(200 -6.6) = 1.6 tribal regions per M agrarian people exposed to nomadic raiding and tribute
        # or ~1/20 of a 31 region tribe per 1M in ag wealth or 1 tribe per 20M additional people over the trigger
        # NOTE: this uses the extremes of nomadic confederation sizes, which might provide an expansion rate early that is too large
        self.confed_tribal_scale = (float(self.confed_max_size - self.confed_min_size)/
                                    float(self.confed_max_population - self.confed_trigger_population)) # tribal regions per M
        print('Outer tribes will expand at 1 tribe per %.2fM additional people' % (self.min_tribal_size/self.confed_tribal_scale))
        geo = self.geography
        steppe_i = [i for i in range(geo.n_regions) if geo.Biome[i] == GEO_STEPPES]
        if len(steppe_i) == 0:
            print('WARNING: No steppe so no nomads in this run!')
            
        self.report_resurrected_polities = False # we deliberately resurrect tribes when confederations fall

        # At this point all the steppe and desert territories are Unoccupied
        # generate a set of territories from the boundaries of the previous tribe
        # else the boundary of the steppe region
        all_territories = np.array(self.territories)
        # first, find all steppe and desert regions within tribal_distance_from_steppe
        min_steppe_distance = self.compute_min_steppe_distance(steppe_i)
        self.trial_data['min_steppe_distance'] = min_steppe_distance # record distances from steppe for all geographical regions

        # BUG: We likely want the initial tribes to be present in Agr2_expand and Agr3_expand regions as well
        # And then as those are converted to agriculture, the tribes need to 'disappear' somehow
        # In the current implementation, we have those areas marked as GEO_AGRICULTURAL but having marginal intensification
        # should really include them.  But this awaits a MASTERS project of adding forest nomads etc that get squeezed in Gaul etc
        # as the biomes are converted.  
        tribal_i = [i for i in range(geo.n_regions) if geo.Biome[i] in [GEO_STEPPES,GEO_DESERT] and min_steppe_distance[i] < self.tribal_distance_from_steppe]
        print('%d steppe regions; %d tribal desert regions within %dkm of steppe' % (len(steppe_i), len(tribal_i), self.tribal_distance_from_steppe))

        # for extortion population reporting under NC
        # For 'exposed, exploitable wealth' reporting purposes
        west_lon = 39 # was Ural_lon but move more west so central (C) reports polities in Persia, etc.
        east_lon = Altai_lon
        # NC strikeable population
        potentially_strikable_territories_i = [i for i in range(geo.n_regions) if min_steppe_distance[i] < self.confed_strike_distance]
        self.western_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and geo.lon[i] < west_lon]
        self.central_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and west_lon <= geo.lon[i] < east_lon]
        self.eastern_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and geo.lon[i] >= east_lon]
        # NT strikeable population
        potentially_strikable_territories_i = [i for i in range(geo.n_regions) if min_steppe_distance[i] < self.tribal_strike_distance]
        self.nt_western_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and geo.lon[i] < west_lon]
        self.nt_central_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and west_lon <= geo.lon[i] < east_lon]
        self.nt_eastern_ag_territories_i = [i for i in potentially_strikable_territories_i if geo.Biome[i] == GEO_AGRICULTURAL and geo.lon[i] >= east_lon]
        print('W: %d/%d C: %d/%d E: %d/%d regions within %.1f/%.1fkm NT/NC strike distance' % (len(self.nt_western_ag_territories_i),len(self.western_ag_territories_i),
                                                                                               len(self.nt_central_ag_territories_i),len(self.central_ag_territories_i),
                                                                                               len(self.nt_eastern_ag_territories_i),len(self.eastern_ag_territories_i),
                                                                                               self.tribal_strike_distance,self.confed_strike_distance))

        # Create the nomadic tribes in the steppe/desert regions
        tribes = self.compute_tribes(tribal_i,list(range(int(self.min_tribal_size),int(self.max_tribal_size))))
        assigned = np.zeros(geo.n_regions,dtype=int ) 
        for tribal_id in unique(tribes):
            if tribal_id <= 0:
                continue
            tribe_i = np.where(tribes == tribal_id)[0]
            # assert population as all elite, no population
            # assert birthrate as zero
            tribe = NomadicTribe(territories=all_territories[tribe_i])
            self.nomadic_tribes.append(tribe)
            assigned[tribe_i] = tribe.id # restate id
            all_lons = [t.lon for t in tribe.territories]
            # These are invariant
            tribe.min_lon = min(all_lons) 
            tribe.max_lon = max(all_lons) 
        # Save this data for (matlab) display purposes
        self.trial_data['nomadic_tribes'] = assigned
        n_tribes = len(self.nomadic_tribes)

        # setup for mounted cavalry warfare diffusion
        # what we know is that it seems to have started in the Pontic-Caspian region 1000BCE at a level to enable NC formation
        # and it diffused to China by 400BCE.  Nomadic confederations formed in different regions a bit later than that
        # (600BCE for Persia (Skythia) and 200BCE for China (Xiongnu))
        # Here we implement diffusion as a longitudinal spreading at a rate between Pontic_Caspian_lon and China_steppe_border_lon
        # in the required number of years BUT in both east and west directions from Pontic_Caspian_lon, west and east
        # We compute the diffusion rate (in degrees/time_step) and then, during the simulation, advance a west and east longitudinal
        # value and select remaining NTs whose 'capitals' are within the bounds and set has_horse_cavalry.  This scheme ignores
        # any latitude speed (assuming instant).  However, it works well with other theaters like Asia that does not include the
        # PC region explicitly since eventually the (eastern) longitudinal boundary will sweep over the steppe at the proper time.
        # This is roughly .13deg/year (~13km/year)

        # We think horse cavalry would be invented *and applied * first in one or more of these tribes
        # See self.mounted_warfare_invention for other constraints
        self.border_NT = [] # NTs with agriculture within NT strike distant;
        # in degrees/year
        self.mounted_warfare_diffusion_rate = abs(China_steppe_border_lon - Pontic_Caspian_lon)/(float(self.mounted_warfare_diffusion_time)/self.time_step)
        self.mounted_warfare_diffusion_frontier = copy.copy(self.nomadic_tribes)
        self.mounted_warfare_invented = False
        
        # To permit finding adjacent tribes that could join a confedeartion we
        # precompute which tribes are connected to each.  Since tribes never
        # annex territory these connections are constant This also allows us to
        # precompute each tribe's set of agrarian territories within striking
        # distance Since Biomes don't change (just change intensification over
        # time) we can precompute the strikable territories even if initially
        # some are not worth it for both agrarian and nomadic tribes
        for tribe in self.nomadic_tribes:
            tribe.s = self.min_polity_s # setup effective starting s at lowest
            # compute nearby 'connected' tribes
            tribes = unique([t.polity for t in tribe.find_annexable()])
            #DEAD py3 tribes = intersect(tribes,self.nomadic_tribes)
            tribes = list(set(tribes) & set(self.nomadic_tribes))
            tribes.sort(key=lambda t:t.id) # make canonical just in case
            # DEAD tribes = filter(lambda t:t not in [self.Hinterland,self.Unoccupied],tribes)
            tribe.tribal_connections = tribes
            # DEBUG print ' %s: %s' % (tribe, tribes)
            # We have to recompute min_steppe_distance from each tribe independently to get that tribe's strikeable regions
            # This is expensive but done once per tribe
            # TODO given a geography file we could compute the distance to all AG from each steppe/desert region once and cache (in create_geo?) {steppe_i -> [n_regions distance]}
            # Then simply load and use the list of tribal steppe regions and find the ones within distances and cache locally on tribe as usual below, then unload
            min_steppe_distance = self.compute_min_steppe_distance(tribe.territories_i)
            # TODO this part might be faster if we used np.where()[0] rather than filter
            # ADSM works by assigning GEO_AGRICULTURAL once and changing Intensification as time goes on
            # Thus all GEO_AGRICULTURAL are set and we can precompute once the set of territories within striking distance
            ag_i = [i for i in range(geo.n_regions) if geo.Biome[i] == GEO_AGRICULTURAL and min_steppe_distance[i] < self.tribal_strike_distance]
            if len(ag_i):
                tribe.ag_territories_striking_distance_NT = (np.array(self.territories)[ag_i]).tolist()
                self.border_NT.append(tribe)
            # And if this tribe becomes part of an NC, what territories can they harrass?
            ag_i = [i for i in range(geo.n_regions) if geo.Biome[i] == GEO_AGRICULTURAL and min_steppe_distance[i] < self.confed_strike_distance]
            if len(ag_i):
                tribe.ag_territories_striking_distance_NC = (np.array(self.territories)[ag_i]).tolist()

        report = 'Created %d tribes; horse cavalry could be invented in %d tribes; diffusion across %.1fdegrees longitude (%.1fkm) at %.4f degrees/year'
        print(report % (n_tribes, len(self.border_NT),
                        (China_steppe_border_lon - Pontic_Caspian_lon),
                        horse_cavalry_diffusion_distance,
                        self.mounted_warfare_diffusion_rate/self.time_step))
        # can't del assigned, min_steppe_distance because used in nested scope
        del all_territories
        # Finished initializing nomadic tribes

    def setup_for_trial(self):
        # Modify global parameters for each trial here, if any, using self.trial to select
        # You can also update self.n_trials to increase or decrease the number of trials you want to run
        # If we do this way rather than stopping in the debugger and asserting; things run MUCH faster!!
        try:
            if False:
                # self.use_regional_k = {0: 0, 1: 1, 2: 2}[self.trial] # increase application of our intensification schedule
                # self.n_trials = 2+1
                
                self.use_regional_k = {0: 2, 1: 3}[self.trial] # compare an earlier intensification schedule, before saturation
                self.n_trials = 1+1

            if False:
                self.faction_death_vs_tolerance = {0: 1.0, 1: 0.5, 2: 0.0}[self.trial]
                self.n_trials = 2+1

        except KeyError:
            raise RuntimeError("Trial %d: Unable to determine some parameter!" % self.trial)

        # ------------------------

        # a constant for 1 region away battles
        # given the units of s we can just compute using geo.interregion_distance_km/geo.interregion_distance_km

        self.unique_lats = unique(self.geography.lat)
        self.s = self.min_polity_s

        # Setup global helper arrays before creating polities so they can be accessed
        n_t = len(self.territories)
        # to rapidly compute distances between capitals and nether regions via numpy vector methods
        self.latitude  = np.zeros(n_t,dtype=float )
        self.longitude = np.zeros(n_t,dtype=float )

        # For help with faction assignments during collapse
        self.collapse_factions = np.zeros(n_t,dtype=int )
        self.collapse_history = {} # has to track factional collapses
        self.collapse_roots = [] # the roots of collapse history trees


        # Do this here so all the arrays above are setup
##s ComparativeHx.py:CompareWithHistory:setup_for_trial
##s PopElites.py:ElitePopulation:setup_for_trial
        # setup these tracking variables
        n_t = len(self.territories)
        self.elites     = np.zeros(n_t,dtype=float ) # the portion that is elite entourage
        self.peasants   = np.zeros(n_t,dtype=float ) # the portion that is peasant
        self.elite_k_fraction = np.ones(n_t,dtype=float )*self.base_elite_k_fraction # what non-zero fraction of k goes to elites initially
        # cache the current expected (diminishing) return or opportunity for each class
        # this is a number between 1.0 (no population on some k) to a possibly large negative number (when a famine occurs, say)
        # most of the time it is between .6 and 0...
        self.elite_opportunity   = np.ones(n_t,dtype=float ) # initially at maximum
        self.peasant_opportunity = np.ones(n_t,dtype=float ) # initially at maximum

##s Population.py:Population:setup_for_trial
        # Do these calculations before Unoccupied and Hinterland are established
        # so the globals are setup for reference by polities
        self.report_regional_population = self.use_regional_k
        intensification_year = self.intensification_year
        intensification_improvement = self.intensification_improvement
        # if this is already set, skip setting up here
        if len(list(self.regional_intensifications.keys())) == 0:
            if self.use_regional_k:
                print('Using region K(%d)' % self.use_regional_k)
                x2 = 2
                agr_m = ('Agr_Mountains',0.05)
                self.regional_intensifications = {-1500: [('Agr_Europe', 1.0),
                                                          ('Agr_Asia', 1.0),
                                                          ('Agr_India', 1.0),
                                                          ('Agr_Mesopotamia', 0.6),
                                                          ('Agr_Africa', 0.66),
                                                          # order matters: these overlap Agr_Europe so these are set lower last
                                                          ('Agr_expand1', 0.5),
                                                          ('Agr_expand2', 0.5),
                                                          # Details on east Asian improvements; order is important in case you use MYRV as 'shape' since it overlaps w/ PearlRV
                                                          ('Agr_LiaoRV', 0.5),
                                                          ('Agr_PearlRV', 0.5),
                                                          ('Agr_MYRV', 1.0),# must be after PearlRV (should intersect w/ Agr_expand1)
                                                          ('Agr_MekongRV',0.5),
                                                          ('Agr_BurmaRV',0.5),
                                                          agr_m], 
                                                  # TODO after every assertion we should reassert Agr_Mountains
                                                  -200: [('Agr_PearlRV', 1.0),agr_m], # Will also 'update' part of MYRV is shape
                                                  0: [('Agr_MekongRV',1.0),agr_m],
                                                  # TODO delay this opening until 500?
                                                  300: [('Agr_expand1', 0.65),agr_m], # turns Japan on...
                                                  400: [('Agr_LiaoRV',0.65),agr_m], # already turned on via expand1... (could drop)
                                                  600: [('Agr_BurmaRV',1.00),agr_m],
                                                  # Medieval doubling in certain locations (including now all of Asia)
                                                  1000: [('Agr_Europe', x2*1.0),
                                                         ('Agr_Asia', x2*1.0), # All of Asia, including the river valleys above
                                                         ('Agr_India', x2*1.0),
                                                         ('Agr_Mesopotamia', 0.6), # No change in Agr_Mesopotamia
                                                         ('Agr_Africa', x2*0.66),
                                                         ('Agr_expand1', 0.65), # No change in Agr_expand1
                                                         # Open the Rus, southern Africa, Kamchatka, etc.
                                                         ('Agr_expand2', 0.65),
                                                         agr_m
                                                         ]
                                                  }
            else:
                print('Using uniform K')
        # print 'Regional intensifications: %s' % self.regional_intensifications # DEBUG

        # ONCE
        # setup numpy global arrays and indexing schemes
        n_t = len(self.territories)
        # These are the current assignments per territory
        # their values can change as polity types change or dxm methods run

        # ODDITY/BUG these calls create np arrays of the proper type BUT
        # if you (or pdb) try to print them you get
        # *** AttributeError: 'NoneType' object has no attribute 'logical_or'
        # In pdb if you try p self.k it does not complain but shows nothing (as it does for None)
        # type(self.k) shows an numpy.ndarray and len(self.k) reports n_t
        # Possible problem in pdb?  Works fine in ipython

        # reflect a climate/biome difference; see setup_for_trial() per territory
        
        # NOTE: The entire world (not just ag) is normally intensified
        self.intensification_factor = np.ones(n_t,dtype=float ) # biome/crop/climate boost (over ME)
        
        # These are the current values per territory depending on polity and history
        self.population = np.zeros(n_t,dtype=float ) # see territory.set_polity() for initialization
        # see polity.update_arable_k() for changes
        self.k = np.ones(n_t,dtype=float ) # ensure always positive to avoid divide-by-zero
        self.birth_rate = np.zeros(n_t,dtype=float )

        # Scale to the effective birth per time step BUT NOT migration fraction since that is a per time step AMOUNT, not a rate!
        self.state_birth_rate = self.base_birth_rate*self.time_step 
        self.hinterland_birth_rate = self.state_birth_rate/self.hinterland_birth_rate_factor

        # TODO at this point self.Hinterland and self.Unoccupied are set up
        #DEBUG print('Popuation density ratio: %.1f\n' % (self.state_density/self.hinterland_density))

        # CRITICAL: Must initialize the intensification factor before we compute any initial population
        # which happens when we set_polity() for Hinterland in super().setup_for_trial() below
        # This is important since we might base population of starting states on surrounding strength of hinterland
        # Hinterland has not been created here so don't update density here
        self.update_intensification(self.first_year,announce=False,adjust_hinterland=False)
        if self.report_regional_population:
            header = 'Population: Year K Total'
            report_i = [] # DEBUG which territories are NOT 'residual'?
            for region in self.region_list:
                region_i = self.region_territories_i[region]
                report_i.extend(region_i)
                header ='%s %s(%d)' % (header,region,len(region_i))
            print('%s Residual' % header)

        # CRITICAL must do this after setting parameters above because this creates Unoccupied and Hinterland
        # which are created using DemographicPolity() which wants to cache them
        # Of course we reset their properties just below
##s WHM.py:WorldHistoricalModel:setup_for_trial
        self.tracking_biomes = biome_types(self.track_polity_biomes) # so saved data know...
        self.this_year = self.first_year
        self.this_pretty_year = pretty_year(self.this_year)
        if isinstance(self.time_step, int) :
            self.years = range(self.first_year,
                               self.last_year+self.time_step,
                               self.time_step)
        else:
            self.years = np.arange(self.first_year,
                                   self.last_year+self.time_step,
                                   self.time_step)
        print('Running from %s to %s in %.1f year increments' % (pretty_year(self.first_year),
                                                                 pretty_year(self.last_year),
                                                                 self.time_step))
        self.random_seed = self.options.seed
        if self.using_mp:
            self.random_seed = None # force recomputation
        if self.random_seed:
            if use_set_versions:
                print('WARNING: Using set functions but setting random seed: Trial will not replicate!')
            seed = self.random_seed
        else: # None
            # explicitly set and report it so we can repeat any bugs we find
            # HACK! delay some number of seconds so the random number seed is not the same in each instance
            time.sleep(self.trial)
            seed = int(time.time())
        # NOTE: Any calls to random before this are not impacted
        # See, for example, the polity flag generator which is initialized on load below
        # BUG: even after we set this, with A2, subsequent runs or trials are NOT identical
        # WHY??? (use of set())
        random.seed(seed)
        print('%s random seed: %d (%f)' % ('Assigned' if self.random_seed else 'Computed', seed,urandom_random()))
        self.random_seed = seed # record 
        
        # DEAD self.area_km2 = np.ones(len(self.territories),dtype=float ) # area of the territory km^2 (initially assume 1km2 per region)
        global polity_counter
        polity_counter = 1; # reset

        for territory in self.territories:
            territory.setup_for_trial()
        self.polities = list() # preserve order of creation always
        # restart the generator before we create polities
        # but after we set the seed so flags will be the same
        if True:
            # several of the markers like '+' and 'x' do not show up well
            open_markers = 'o^v<>ph' # only 'open' markers (not d which looks thin)
            self.flag_generator = polity_flag(markers=open_markers) # initialize the generator
        else:
            self.flag_generator = polity_flag() # initialize the generator with all markers
        # Initialize the starting non-polities
        self.Unoccupied = self.PolityType(name='Unoccupied')
        self.Unoccupied.make_quasi_state(arising_polity_type=None) # explicitly None here
        self.Unoccupied.flag = {'marker':'.','color':'Silver'} # overwrite flag
        self.Hinterland = self.PolityType(name='Hinterland')
        self.Hinterland.make_quasi_state(arising_polity_type=self.PolityType)
        self.Hinterland.flag = {'marker':'.','color':'Green'} # overwrite flag
        self.create_states = None
        # TODO supers().setup_for_trial() here? so setup_for_trial() methods can complete
        # TODO move print_scalar_members() call to setup_chronicler_for_trial() and add to trail_data 
        self.setup_chronicler_for_trial()


    # Where we actually run a world in a trial
    # These methods are the heart of the simulation and are common to all specializations
    # which override or extend (via super()) inherited behavior
##e WHM.py:WorldHistoricalModel:setup_for_trial
##< Population.py:Population:setup_for_trial



        # CRITICAL *reset* k and br once for these special polities
        self.Unoccupied.density = self.unoccupied_density
        self.Unoccupied.br = 0 # no growth with this 'polity'
        self.Hinterland.density = self.hinterland_density
        self.Hinterland.br = self.hinterland_birth_rate

##e Population.py:Population:setup_for_trial
##< PopElites.py:ElitePopulation:setup_for_trial


##e PopElites.py:ElitePopulation:setup_for_trial
##< ComparativeHx.py:CompareWithHistory:setup_for_trial

        # Always set up accumulators, etc. in case comparison might occur
        self.hx_times = {500:  {'marker':'.', 'color':'b'},
                         1000: {'marker':'s', 'color':'r'},
                         3000: {'marker':'d', 'color':'k'},
                         }
        # The number of regions is fixed after self.load_geography()
        # This is the 'predicted' counter part of geogrraphy.history
        global dump_predicted_history
        if dump_predicted_history:
            self.predicted_history = np.zeros((self.geography.n_regions,len(self.geography.years)),dtype=int )
            self.trial_data['predicted_history'] = self.predicted_history
        
        self.predictions = np.zeros(self.geography.n_regions,dtype=int )
        # setup actuals and prediction arrays
        self.hx_actuals = {} # accumulators for observables
        self.hx_predictions = {} # accumulators for predictions
        self.hx_years = {}
        for interval in list(self.hx_times.keys()):
            # deliberately float and not int so scaling and detrending don't drop counts
            self.hx_actuals[interval]     = np.zeros(self.geography.n_regions,dtype=float )
            self.hx_predictions[interval] = np.zeros(self.geography.n_regions,dtype=float )
            self.hx_years[interval] = int(self.first_year) - interval

        global dump_R2_data
        if dump_R2_data:
            self.trial_data['R2_date'] = []
            self.trial_data['R2_interval'] = []
            self.trial_data['R2_results'] = []
            if dump_R2_observations:
                self.trial_data['R2_observations'] = []
                self.trial_data['R2_predictions'] = []

##e ComparativeHx.py:CompareWithHistory:setup_for_trial

        if self.confed_strike_distance_scale:
            self.confed_strike_distance = self.confed_strike_distance_scale*self.tribal_strike_distance

        # given time and assumed birthrate and starting misery threshold, what is the level of faction death and increased misery threshold
        # logistic_P(0.8,1.0,0.0175,30) = 0.8712
        # logistic_P(0.9,1.0,0.0175,30) = 0.9383
        # ADSM eqn 9
        self.increased_misery_threshold = logistic_P(self.base_misery_threshold, 1.0, self.base_birth_rate,
                                                     self.faction_increased_asabiya_time*(1 - self.faction_death_vs_tolerance))
        # Compute how much haircut to give a faction's elites so that have running room
        # how many elites must die within a faction to given it enough opportunity to avoid rebellion in faction_increased_asabiya_time years
        # ADSM eqn 10
        self.collapse_elite_population_fraction = logistic_P0(self.base_misery_threshold,1.0,self.base_birth_rate,
                                                              self.faction_increased_asabiya_time*self.faction_death_vs_tolerance)

        self.sea_battle_distance = 0
        self.sea_distance_factor = 1 # no shortening
        self.desert_battle_distance = 0
        self.desert_distance_factor = 1 # no shortening
        # for flushing border caches and recomputing distances
        self.sea_battle_distance_last = self.sea_battle_distance
        self.desert_battle_distance_last = self.desert_battle_distance

        self.update_power_projection()
        
        # Summarize status based on final assignment of parameters above
        print('Trial %d tracking states %d regions or greater in %s biomes' % (self.trial, self.compare_polity_size, self.tracking_biomes))
        initial_s = self.s
        print("Trial %d projection factor: %.2f victory: %.2f" % (self.trial, initial_s, self.War_victory))
        # Ensure there will be some chance to assign territories to factions
        self.faction_add_odds = max(0.1, self.faction_add_odds)
        print("Trial %d faction grace period: %.1f years (%.1f%% death/%.1f%% tolerance); %.1f%% chance of passing" % (self.trial, self.faction_increased_asabiya_time,
                                                                                                                       self.faction_death_vs_tolerance*100.0,
                                                                                                                       (1 - self.faction_death_vs_tolerance)*100.0,
                                                                                                                       (1 - self.faction_add_odds)*100.0))
        print("Trial %d Collapse misery threshold: %.3f, post-collapse misery threshold: %.3f" % (self.trial, self.base_misery_threshold,self.increased_misery_threshold))
        print("Trial %d migration rates: peasants: %.4f/year elites: %.4f/year" % (self.trial, self.migration_fraction_peasants/self.time_step,self.migration_fraction_elites/self.time_step))
        n = float(len(self.faction_odds))
        for i in np.sort(unique(self.faction_odds)):
            c = float(self.faction_odds.count(i))
            print(" %.2f%% chance to collapse to %d factions" % (c*100.0/n,i))

        self.setup_regions_for_trial()
        self.setup_nomads_for_trial()
        
        # Report DXM instructions
        print('State creation instructions:')
        pprint.pprint(self.create_states_instructions)
        print('Intensification changes:')
        pprint.pprint(self.regional_intensifications)
        print('Keep sea connections:')
        pprint.pprint(self.keep_sea_connections)
        print('Sea battle distance changes: (+%.3fkm/yr)' % self.sea_battle_distance_increment)
        pprint.pprint(self.sea_distance_schedule)
        print('Keep desert connections:')
        pprint.pprint(self.keep_desert_connections)
        print('Desert battle distance changes: (+%.3fkm/yr)' % self.desert_battle_distance_increment)
        pprint.pprint(self.desert_distance_schedule)
        # BUG do we know the rate yet??
        print('Invent horse cavalry: %s spreading %.1fkm in %d years' % (self.mounted_warfare_invention,
                                                                         horse_cavalry_diffusion_distance,
                                                                         self.mounted_warfare_diffusion_time))
        
    def possibly_create_new_states(self):
        global whm
        self.create_states = None
        try:
            # Are there any instructions for creating states by fiat in this_year?
            # See NADSMX for examples
            instructions = self.create_states_instructions[self.this_year]
            self.create_states = []
            geo = self.geography
            Hinterland = self.Hinterland
            for instruction in instructions:
                spatial_defn,sizes,polity_type = instruction
                region_ids = []
                # TODO support union of several spatial defns, including a mix of bb and states
                if len(spatial_defn) == 2:
                    state_name,year = spatial_defn
                    try:
                        p_id = geo.empire_names.index(state_name)
                    except ValueError:
                        # this must be a (sub) geopgraphy that does not include this state
                        print('Missing %s -- skipping making states there' % state_name)
                        continue
                    if np.isnan(year):
                        region_ids = np.nonzero(geo.history == p_id)[0]
                    else:
                        year_i = geo.years.index(year)
                        region_ids = np.nonzero(geo.history[:,year_i] == p_id)[0]
                else:
                    # min_lon,min_lat,max_lon,max_lat = spatial_defn
                    min_lat,min_lon,max_lat,max_lon = spatial_defn
                    region_ids = [i for i in range(geo.n_regions) if min_lon <= geo.lon[i] <= max_lon and min_lat <= geo.lat[i] <= max_lat]

                region_ids = unique(region_ids)
                region_ids = [t for t in region_ids if self.territories[t].Biome == GEO_AGRICULTURAL and self.territories[t].polity is Hinterland]
                n_states = len(self.create_states)
                if not isinstance(sizes, list): # if they specify np.nan we generate one state
                    # Use entire list to create a single state
                    self.create_states.append((polity_type,region_ids))
                else:
                    # Generate a set of small states of various sizes out of the given regions
                    small_states = self.compute_tribes(region_ids,sizes)
                    for state_id in unique(small_states):
                        if state_id <= 0:
                            continue
                        state_i = np.where(small_states == state_id)[0]
                        self.create_states.append((polity_type,state_i)) # See Population.deus_ex_machina() (via super()) for actual creation
                
                print("%s: Creating %d states from %d regions: %s,%s" % (self.this_pretty_year,
                                                                         len(self.create_states) - n_states,
                                                                         len(region_ids), spatial_defn, sizes))
            self.create_initial_states_from_history = False
        except KeyError:
            pass

    def possibly_update_long_distance_efficiencies(self):
        global whm
        # increase sea and desert attack distance
        # TODO if whm.enable_border_caching flush all extant polity caches if the distance since last checkpoint
        # has grown by geo.interregion_distance_km
        self.sea_battle_distance += self.sea_battle_distance_increment
        self.desert_battle_distance += self.desert_battle_distance_increment
        # If the distance discount factors change then so do the underlying cached distances from capitals
        distance_factors_changed = False
        try:
            distance_factor = self.sea_distance_schedule[self.this_year]
            self.sea_battle_distance = distance_factor[0]
            new_distance_factor = distance_factor[1]
            distance_factors_changed |= (new_distance_factor != self.sea_distance_factor)
            self.sea_distance_factor = new_distance_factor
            print('%s: Updating sea distance to %.1fkm with distance factor %.3f' % (self.this_pretty_year,
                                                                                     self.sea_battle_distance,
                                                                                     self.sea_distance_factor))
        except KeyError:
            pass
        try:
            distance_factor = self.desert_distance_schedule[self.this_year]
            self.desert_battle_distance = distance_factor[0]
            new_distance_factor = distance_factor[1]
            distance_factors_changed |= (new_distance_factor != self.desert_distance_factor)
            self.desert_distance_factor = new_distance_factor
            print('%s: Updating desert distance to %.1fkm with distance factor %.3f' % (self.this_pretty_year,
                                                                                        self.desert_battle_distance,
                                                                                        self.desert_distance_factor))
        except KeyError:
            pass

        # DEAD border_dist = whm.geography.interregion_distance_km
        border_dist = 100 # km: don't update until there as been a change in distance by at least this amount (keeps computation down)
        if (distance_factors_changed or
            (self.sea_battle_distance    >= self.sea_battle_distance_last    + border_dist) or 
            (self.desert_battle_distance >= self.desert_battle_distance_last + border_dist)):
            global enable_border_caching # from WHM
            if distance_factors_changed:
                print('%s: Distance factors changed!' % self.this_pretty_year)
            if enable_border_caching or distance_factors_changed:
                for polity in self.polities:
                    if not polity.quasi_state and polity.polity_type == POLITY_TYPE_AG and polity.state is POLITY_ALIVE:
                        # DEBUG print '  Updating distance costs for %s from %.1fN, %.1fE' % (polity, polity.Clat, polity.Clon)
                        polity.set_border_cache()
                        if distance_factors_changed:
                            polity.update_distance_from_capital() # This is expensive
            # Update for next time...
            self.sea_battle_distance_last = self.sea_battle_distance
            self.desert_battle_distance_last = self.desert_battle_distance

    def possibly_invent_and_diffuse_nomadic_horse_cavalry(self):
        global whm
        # Implement nomadic mounted warfare DIFFUSION (vs. s diffusion via states)
        if self.mounted_warfare_invented:
            if len(self.mounted_warfare_diffusion_frontier) > 0:
                # Initially mounted_warfare_diffusion_frontier contains all NT which have not acquired horse_cavalry
                # We remove them incrementally below
                self.western_cavalry_diffusion_limit -= self.mounted_warfare_diffusion_rate
                self.eastern_cavalry_diffusion_limit += self.mounted_warfare_diffusion_rate
                # Cheap test to use the 'capital' of the tribe as the trigger rather than any territory covered.
                # This mean there is a slight delay in starting diffusion.
                mw_tribes = [nt for nt in self.mounted_warfare_diffusion_frontier if (self.western_cavalry_diffusion_limit <= nt.max_lon and
                                                                                      self.eastern_cavalry_diffusion_limit >= nt.min_lon)]
                for tribe in mw_tribes:
                    tribe.has_horse_cavalry = self.this_year
                    self.mounted_warfare_diffusion_frontier.remove(tribe)
                    print('%s: Diffusing mounted warfare to %s at %dE' % (self.this_pretty_year,tribe,tribe.Clon))
                if len(self.mounted_warfare_diffusion_frontier) == 0: # trailing edge trigger
                    self.mounted_warfare_diffusion_end_year = self.this_year
                    print('Finished diffusing mounted warfare in %s' % self.this_pretty_year)
        else:
            # Not yet invented...unpack the invention specification until HC is invented
            W_lon, E_lon, start_year, start_popM = self.mounted_warfare_invention
            possible_inventors = self.border_NT
            if start_popM is not None:
                inventors = []
                for nt in possible_inventors:
                    ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = nt.ag_polities_within_striking_distance()
                    if total_extortable_population/million >= start_popM:
                        inventors.append(nt)
                possible_inventors = inventors
            if start_year is not None and start_year > self.this_year:
                pass
            else:
                # the year is 'right' and we have a set of possible_inventors (could be empty)
                if W_lon is not None: 
                    possible_inventors = [nt for nt in possible_inventors if W_lon <= nt.max_lon and E_lon >= nt.min_lon]
                    if len(possible_inventors):
                        self.mounted_warfare_invented = True
                        self.western_cavalry_diffusion_limit = np.mean([nt.min_lon for nt in possible_inventors])
                        self.eastern_cavalry_diffusion_limit = np.mean([nt.max_lon for nt in possible_inventors])
                    elif start_year is not None and start_popM is None:
                        # We are in a geography like Asia and we just have a year and W/E lon location
                        # There will be no inventing tribes (because they are 'off the map') but diffusion
                        # starts nevertheless.  This was the original behavior
                        self.mounted_warfare_invented = True
                        self.western_cavalry_diffusion_limit = W_lon
                        self.eastern_cavalry_diffusion_limit = E_lon
                    else:
                        # We have a spreading location but no start date or population trigger
                        # We will *never* trigger diffusion
                        pass
                else:
                    # no geographical restriction
                    if len(possible_inventors):
                        self.mounted_warfare_invented = True
                        self.western_cavalry_diffusion_limit = np.mean([nt.min_lon for nt in possible_inventors])
                        self.eastern_cavalry_diffusion_limit = np.mean([nt.max_lon for nt in possible_inventors])

                if self.mounted_warfare_invented:
                    self.mounted_warfare_diffusion_start_year = self.this_year
                    print('Inventing horse cavalry in %s; spreading center: %.1fE to %.1fE:' % (pretty_year(self.mounted_warfare_diffusion_start_year),
                                                                                                self.western_cavalry_diffusion_limit,
                                                                                                self.eastern_cavalry_diffusion_limit))
                    for nt in self.border_NT: # all possible border inventions
                        n_connections = len(nt.tribal_connections)
                        if True or n_connections > 0: # if you don't connect you don't get to seed an NC
                            ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = nt.ag_polities_within_striking_distance()
                            print(' %s%s (%dN, %dE) %d/%d: Extortable population: %.2fM' % (('>>' if nt in possible_inventors else '  '),
                                                                                            nt,nt.Clat,nt.Clon,nt.n_territories,n_connections,
                                                                                            total_extortable_population/million))

    def deus_ex_machina(self):
        global whm

        self.possibly_create_new_states()
        
##s Population.py:Population:deus_ex_machina
##s WHM.py:WorldHistoricalModel:deus_ex_machina
        # Acts of 'god': possibly over multiple territories in this year;
        # Most of the time this is a pass.
        pass

##e WHM.py:WorldHistoricalModel:deus_ex_machina
##< Population.py:Population:deus_ex_machina

        
        # If actual history we want to use the initial states and hinterland distribution
        # to discover a hinterland density, then scaled, to match the observed total population
        # We then set_population NOT density and let the parameteric density take over
        # This ensures that the population estimate matches the observed population at the start

        # ADSM also uses this code to get an initial population start
        # However, it has an additional constraint that the local power around the state has to be large enough
        # to ensure it doesn't collapse right away.

        # There are three cases:
        # PNAS  - actual history, actual population, no initial creation of polities (create_geography with PNAS tag)
        # ADSM1 - actual history, actual population, initial creation of polities from create_geography
        # ADSM2 - fake   history, fake   population, initial creation of polities from create_arena
        
        if self.this_year == self.first_year:
            geography = self.geography
            year_i = geography.years.index(self.this_year)

            # find the regions/territories that were occupied in this year
            Hinterland = self.Hinterland
            hinterland_i = Hinterland.territories_i
            # first, assign population to Ag hinterland regions
            hint_if_sum = np.sum(self.intensification_factor[hinterland_i]*self.area_km2[hinterland_i])
            # Assume starting population needs to be divided appropropriately between hinterland and states
            if geography.actual_history:
                # Use actual population from history
                total_population = geography.total_population[year_i]
            else:
                # Use a population based on initial population density
                # BUG really should use a bottom up estimate using the main densities and then scale back using initial_population_scale
                # if scale is 1, however, states  will pretty-much immediately collapse under ADSM so careful
                # The initial density (1.45 typically) is < 4 (normal hinterland density) so it starts about about 50% burden
                # which is then reduced further by initial_population_scale (.2*1.45) == .29p/km2
                # typical initial population per GEO_AGRICULTURAL region is 28K
                total_population = np.sum(self.population)
            total_population *= self.initial_population_scale
            density = total_population/hint_if_sum
            print('Initialization hinterland density (%s/%s km^2) = %.2f/km^2' % (millions(total_population), millions(hint_if_sum), density))
            self.initial_hinterland_density = density # record this
            # TODO ensure density is between hinterland and state
            self.set_population(Hinterland,density,hinterland_i)

            if self.create_states is None and self.create_initial_states_from_history:
                history = geography.history
                # Find all the occupiable territories of any states this year
                # Could be none...
                # assume states are agrarian and only arise in GEO_AGRICULTURAL biomes, hence Hinterland
                start_history = np.array(history[:,year_i])
                self.create_states = []
                for e_id in unique(start_history):
                    if e_id == 0:
                        continue
                    region_ids = np.where(start_history == e_id)[0]
                    print("%s: Creating %d region state based on %s" % (self.this_pretty_year, len(region_ids),
                                                                        geography.empire_names[e_id]))
                    self.create_states.append((self.PolityType,region_ids))
            else:
                print('Not creating initial states based on history!')

        if self.create_states is not None:
            k_scale = self.state_density/self.hinterland_density
            all_territories = np.array(self.territories)
            for emp_tuple_i in self.create_states:
                polity_type,emp_i = emp_tuple_i
                # NOTE: states using typical ancient densities need to have no more than 9 regions or they can be taken over by powerful hinterland forces
                # when Center_large had a 4x4 state rather than a 3x3 with default parameters it was melted away by other smaller hinterland-started
                # states within 30 years.  But starting from 3x3 it builds power (at the right time) and dominates the landscape for 700 years before collapse
                e = polity_type(territories=all_territories[emp_i]) # annex/convert these hinterland cells w/o cost
                self.states_created.add(e.id) # done...
                self.set_population(e,k_scale*self.initial_hinterland_density,emp_i)
            self.create_states = None

        self.update_intensification(self.this_year)

##e Population.py:Population:deus_ex_machina
        # Population.deus_ex_machina() has run so if there were states we would know
        if len(self.states_created) == 0:
            # (N)ADSM has no theory about where or where the first states were formed
            # it does try to explain their the spread though...
            # A later theory ala 'circumscription' in delta regions might allow spontaneous creation
            # but the timing on that depends a lot on pre-history, etc.  We just take it as given and go from there
            raise RuntimeError("No initial states declared for %s.  Unable to continue!" % self.geography.name)

        self.possibly_update_long_distance_efficiencies()
        
        # DEBUG Report extorable population for Pontic-Caspian tribes at different times
        if True and self.this_year in [-1000, -600]: # report the predicted extortable population around the Pontic-Caspian region (+/- 3 degrees lon)
            pc_nt = [nt for nt in self.nomadic_tribes if (Pontic_Caspian_lon - 3)  <= nt.max_lon and (Pontic_Caspian_lon + 3) >= nt.min_lon]
            ag_states = []
            for nt in pc_nt:
                ag_within_reach,total_extortable_population,max_s_polity,max_pop_polity = nt.ag_polities_within_striking_distance()
                ag_states.extend(ag_within_reach)
            total_extortable_population = 0
            ag_states = unique(ag_states) # avoid double counting
            for ag_state in ag_states:
                total_extortable_population += ag_state.total_population
            print('%s: PC extortable population: %s from %d states' % (self.this_pretty_year,
                                                                        millions(total_extortable_population),
                                                                        len(ag_states)))

        self.possibly_invent_and_diffuse_nomadic_horse_cavalry()

        # possible update of global s and perhaps hinterland's assumed power projection
        self.update_power_projection()

    def set_population(self,polity,population_density,territories_i):
##s PopElites.py:ElitePopulation:set_population
##s Population.py:Population:set_population
        self.population[territories_i] = population_density*self.intensification_factor[territories_i]*self.area_km2[territories_i] # ADSM eqn 2
        
##e Population.py:Population:set_population
##< PopElites.py:ElitePopulation:set_population

        # apportion what the super() set between elite and peasants
        pop = self.population[territories_i]
        elite_fraction = self.base_elite_k_fraction
        peasant_fraction = 1.0 - elite_fraction
        self.elites  [territories_i] = pop*elite_fraction
        self.peasants[territories_i] = pop*peasant_fraction

##e PopElites.py:ElitePopulation:set_population
        # update raw_power given new population
        polity.update_power()
        

    def state_of_the_world(self):
##s ComparativeHx.py:CompareWithHistory:state_of_the_world
##s Population.py:Population:state_of_the_world
##s WHM.py:WorldHistoricalModel:state_of_the_world
        pass

##e WHM.py:WorldHistoricalModel:state_of_the_world
##< Population.py:Population:state_of_the_world

        self.update_population() # PopElite overrides

##e Population.py:Population:state_of_the_world
##< ComparativeHx.py:CompareWithHistory:state_of_the_world

        global enable_R2_comparison

        for interval,hx_predictions in self.hx_predictions.items():
            hx_predictions += self.predictions
        self.predictions[:] = 0 # reset

        int_this_year = int(self.this_year)
        if abs(self.this_year - int_this_year) > self.time_step:
            return
        # TODO:
        # The observables are on a 1C time pace so compare on that sequence
        # But the model is running at some other time step and we need to update
        # the accumulation of predictions and clear on comparison demand
        
        # TODO do this AFTER the comparison below?  The historical data we got
        # from Peter encoded, each century, which polity occupied a *region*
        # if the polity controlled a minimum number of regions.  This included
        # nomadic confederations, etc. and thus may or may not be what the model is
        # trying to predict.  Thus we restrict those data to the regions in
        # non-Unoccupied territory, under the assumption that all models use
        # Unoccupied as a flag meaning 'ignore this' for statistics purposes
        try:
            # BUG really only want to do this once but if fractional will do this several times
            century_i = self.geography.years.index(int_this_year)
        except (ValueError, IndexError ):
            pass
        else:
            if enable_R2_comparison:
                # Track actual historical counts
                # NOTE: We use Unoccupied as the proxy for what should be included or not from history
                # Therefore all models should assign Unoccupied to territories that they wish ignored
                # for this century.  Generally this is the case, esp for PNAS which has changing Agr includes

                regions = []
                # this code always computes the same set of regions unless we expect the Biome assignment to change during the run
                # in ADSM we set GEO_AGRICULTURAL to all possible and then modulate their intensity to control 'conversion'
                # In the long run we should have GEO_FOREST and track it properly in Western Europe (and maybe Khmer/south India).
                for territory in self.territories:
                    if territory.Biome not in self.track_polity_biomes:
                        continue # skip this one
                    # BUG Using Unoccupied is a bad idea:  In ADSM at the start we assign this to all non-ag territories
                    # and so if we aren't interested in ag-only we should look everywhere for actual polities
                    # It should be the case that if we are ag only then we'll just focus there even if polities are growing elsewhere.
                    if True: #  or territory.polity is not self.Unoccupied: # DEAD
                        # BUG exclude from regions non-ag if we are ag-only
                        regions.extend(territory.regions)
                # where, in history, of the modelled regions was there actually a macrostate?
                current_history = self.geography.history[regions,century_i] # reduce to regions we are tracking
                polity_present_i = np.where(current_history)[0] # where non-zero (hinterland) polities are
                polity_ids = unique(current_history[polity_present_i]) # 'zero' is skipped since we look at polities ony
                regions = np.array(regions) # convert for indexing
                for p_id in polity_ids:
                    p_regions_i = np.where(current_history[polity_present_i] == p_id)[0]
                    if len(p_regions_i) >= self.compare_polity_size:
                        p_regions_i = regions[polity_present_i[p_regions_i]]
                        for interval,hx_actuals in self.hx_actuals.items():
                            hx_actuals[p_regions_i] += 1

        # see if it is time to compare
        # if we are starting the modulus test succeeds below
        # but we haven't really accumulated data yet do skip this...
        if report_R2_middle and self.this_year == self.first_year:
            return
        for interval,hx_predictions in self.hx_predictions.items():
            if ((int_this_year - self.first_year) % interval) == 0: 
                hx_actuals = self.hx_actuals[interval]

                # compute R2 between hx_actuals (observations) and hx_predictions (predictions)
                observations = copy.copy(hx_actuals) # otherwise we destroy the value with the -= below
                # scale the per-time-step predictions to be per-century counts
                predictions  = hx_predictions/(century/self.time_step) # normalize counts per century (UNNEDED given detrending below?)
                global dump_R2_data
                if dump_R2_data:
                    self.trial_data['R2_date'].append(self.this_year)
                    self.trial_data['R2_interval'].append(interval)
                    if dump_R2_observations:
                        # Save copies of (scaled) counts
                        self.trial_data['R2_predictions'].append(copy.copy(predictions))
                        self.trial_data['R2_observations'].append(copy.copy(observations))
                
                if enable_R2_comparison:
                    delta = predictions - observations # UNUSED
                    predictions  -= np.mean(predictions) # detrend (remove mean)
                    observations -= np.mean(observations) # detrend (remove mean)
                    R2 = 0
                    R2_str = ''
                    sum_predictions2  = np.sum(predictions **2)
                    sum_observations2 = np.sum(observations**2)
                    if sum_predictions2 == 0: # is there any variance around the mean?
                        R2_str = "%s<no prediction variance>" % R2_str
                        if sum_observations2 == 0: # is there any variance around the mean?
                            R2_str = "%s<no polity variance>" % R2_str
                            R2 = 1.0 # both are 0/0 so they match perfectly
                    elif sum_observations2 == 0: # is there any variance around the mean?
                        R2_str = "%s<no polity variance>" % R2_str
                    else:
                        # There was at least some variance around both means
                        R2 = (np.sum(predictions*observations)**2)/(sum_predictions2*sum_observations2)
                        R2_str = '%.3f' % R2
                    # print or plot or both
                    if report_R2_middle:
                        half_interval = interval/2
                        report_year = self.this_year - half_interval;
                    else:
                        report_year = self.this_year
                    if dump_R2_data:
                        self.trial_data['R2_results'].append([interval,report_year,R2])
                    print('R2(%d): %s %s' % (interval,pretty_year(report_year),R2_str)) # DEBUG

                    if self.display_show_figures:
                        marker_info = self.hx_times[interval]
                        # if multiple trials distinguish by color only and trial marker
                        marker = self.trial_marker if self.n_trials > 1 else marker_info['marker']
                        # TODO if report_R2_middle consider a horizontal 'error bar' that indicates the interval (clutter)
                        if True and report_R2_middle:
                            self.display_ax_R2.errorbar(report_year,R2,
                                                        marker=marker,
                                                        xerr=half_interval,capsize=0,
                                                        color=marker_info['color'],
                                                        markersize=6, # Make it large(r) and readable
                                                        label='_nolegend_',
                                                        clip_on=False) # avoid clipping markers on far right
                        else:
                            self.display_ax_R2.plot(report_year,R2,
                                                    marker=marker,
                                                    color=marker_info['color'],
                                                    markersize=6, # Make it large(r) and readable
                                                    label='_nolegend_',
                                                    clip_on=False) # avoid clipping markers on far right
                        self.display_fig_R2.canvas.draw() # force out drawing
                # reset actuals and predictions
                hx_predictions[:] = 0
                hx_actuals[:] = 0
        
        # To make things faster when computing war stuff precompute the per-territory power in the world
        self.raw_territorial_power = self.elites
        # DEBUG
        if False:
            for polity in self.polities:
                if polity.state is POLITY_ALIVE and not polity.quasi_state:
                    power = polity.power_scale[polity.border_territories_i]
                    avg_power = np.exp(-np.sqrt(polity.size/math.pi)/self.s)
                    print('%s: %s %d %.3f %.3f %.3f' % (self.this_pretty_year, polity,
                                                        polity.size,min(power),avg_power,max(power)))


    # Chronicler additions for NADSM
    def setup_chronicler_for_trial(self):
##s PopChronicler.py:PopChronicler:setup_chronicler_for_trial
        self.predicted_population = [] # recorded for each 'display' year
        self.region_population = None
        if report_regional_population and self.geography.years:
            self.region_population = np.zeros((self.geography.n_regions,len(self.geography.years)),dtype=float )
##s Chronicler.py:Chronicler:setup_chronicler_for_trial
        self.display_counter = 0 # initialize for movies
        self.trial_data = {} # initialize dictionary for accumulating data for the eventually mat file

        self.predicted_years = []
        self.predicted_polity_count = []
        self.predicted_cumulative_polities = [] # which were previously alive at the display bound?
        self.predicted_cumulative_polity_count = []
        self.predicted_regions_under_polity = []
        self.predicted_km2_under_polity = []
        self.predicted_ag_regions_under_polity = []
        self.predicted_ag_km2_under_polity = []

        if self.display_show_figures:
            # BUG: this assignment should only happen once, via self.setup_for_trial()
            # However, since Chronicle is mixed in with WHM as a base class
            # the multiple inheritance means only one of the setup methods
            # will be called by super().
            # So instead we have a special setup call for the chronicler
            # TODO: convert trial_markers string to a set of marker instances with fillstyle=None
            self.trial_marker = trial_markers[self.trial % len(trial_markers)]
            print("Trial %d marker: %s" % (self.trial, self.trial_marker))
            self.title_tag = "%s\n%s" % (self.name,self.geography_name)
            #DEAD self.title_tag = "%s %s %d" % (self.name,self.geography_name,self.trial) #DEBUG REMOVE
            self.update_display_for_trial()
        
##e Chronicler.py:Chronicler:setup_chronicler_for_trial
##< PopChronicler.py:PopChronicler:setup_chronicler_for_trial

        
##e PopChronicler.py:PopChronicler:setup_chronicler_for_trial
        global report_annexations
        if report_annexations:
            self.annexation_data = []
        global report_polity_history
        if report_polity_history:
            self.polity_history = [];
            self.collapse_data = []

    def chronicle(self): # runs each time step
##s Chronicler.py:Chronicler:chronicle
        if (self.this_year % self.display_time_interval) == 0:
            self.update_chronicle()
            if self.display_show_figures:
                self.update_display()
            
##e Chronicler.py:Chronicler:chronicle
