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

    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
        super().__init__(name=name,territories=territories)
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
        super().state_of_the_union()
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
        super().set_state(state)
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
        super().__init__(name=name,territories=territories)
        

    def extort(self,polity):
        raise RuntimeError("How can agrarian polity %s be extorting?" % self)

    def set_state(self,state):
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
    def __init__(self,name=None):
        global whm
        super().__init__(name=name)
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
        super().collapse() # WHM version using split

    def state_of_the_union(self):
        global whm
        super().state_of_the_union()
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

    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
        # There are a fixed set of these asserted at the beginning of the run
        # As they are absorbed into confedations the tribes will oscillate between POLITY_ALIVE and POLITY_DEAD
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
    PolityType = AgrarianPolity
    TerritoryType = EliteOccupiedTerritory

    def __init__(self,options):
        options.description = 'Simulate nomadic and agrarian states in a geography'
        # options.add_argument('-m',metavar='model', help='Model name to use',default='NADSM')
        options.add_argument('--disable_nomads',action='store_true',help='Disable nomad formations',default=False)
        options.add_argument('--disable_nc',action='store_true',help='Disable nomad confederation formation',default=False)
        super().__init__(options)

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

        super().establish_territories()

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

        super().initialize_parameters()

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
        super().setup_for_trial()

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
        
        super().deus_ex_machina()
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
        super().set_population(polity,population_density,territories_i)
        # update raw_power given new population
        polity.update_power()
        

    def state_of_the_world(self):
        super().state_of_the_world()
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
        super().setup_chronicler_for_trial()
        global report_annexations
        if report_annexations:
            self.annexation_data = []
        global report_polity_history
        if report_polity_history:
            self.polity_history = [];
            self.collapse_data = []

    def chronicle(self): # runs each time step
        super().chronicle()
        global report_polity_history
        if report_polity_history:
            for polity in self.polities:
                if not polity.quasi_state and polity.size and polity.state is POLITY_ALIVE: # alive with territories
                    territories_i = polity.territories_i;
                    elite_k = np.sum(self.k[territories_i]*self.elite_k_fraction[territories_i])
                    elite_pop = np.sum(self.elites[territories_i])
                    if polity.polity_type == POLITY_TYPE_AG:
                        min_elite_opportunity_i = territories_i[self.elite_opportunity[territories_i].argmin()];
                        min_elite_opportunity = self.elite_opportunity[min_elite_opportunity_i]
                    elif polity.polity_type == POLITY_TYPE_NT:
                        continue # skip these guys
                    elif polity.polity_type == POLITY_TYPE_NC:
                        min_elite_opportunity = 0 # No elite opportunity maintained for NCs
                    self.polity_history.append([polity.id,self.this_year,polity.size,polity.total_population,polity.total_k,elite_pop,
                                                elite_k,polity.s,min_elite_opportunity,polity.km2_size])

    
    def update_chronicle(self):
        if True: # DEBUG Report regional population within strike zones of powered up NT and NCs
            territories = self.territories
            def region_population(ag_territories_i):
                population = 0
                for polity in unique([territories[i].polity for i in ag_territories_i]):
                    if not polity.quasi_state:
                        population += polity.total_population # total, not fraction, since this is total wealth
                return millions(population)
                
            print('%s W: %s/%s C: %s/%s E: %s/%s' % (self.this_pretty_year,
                                                     region_population(self.nt_western_ag_territories_i),region_population(self.western_ag_territories_i),
                                                     region_population(self.nt_central_ag_territories_i),region_population(self.central_ag_territories_i),
                                                     region_population(self.nt_eastern_ag_territories_i),region_population(self.eastern_ag_territories_i)))
        super().update_chronicle()

    def update_display_figure_hook(self, axis):
        super().update_display_figure_hook(axis)
        if axis == self.display_ax_p:
            if False: # DEBUG
                print("Battle distance: sea %.2fkm; desert %.2fkm" % (self.sea_battle_distance,self.desert_battle_distance))
            for polity in self.polities:
                if polity.state is POLITY_ALIVE and polity.large_enough() and polity.Clon is not np.nan:
                    # TODO ignore large_enough so even small but living states are shown?
                    # show location of capital
                    self.display_ax_p.plot(polity.Clon,polity.Clat,
                                           marker='.',linestyle='None',color='Blue')
                    if polity.marked_polity():
                        self.display_ax_p.text(polity.Clon,polity.Clat,polity.id,color='Blue')
                    if False:
                        # DEBUG how close to critical are we?
                        print(" %s (%s/%s) %d %d %.6f" % (polity,polity.Clon,polity.Clat,
                                                          len(polity.territories),len(polity.border_territories),
                                                          min(self.elite_opportunity[polity.territories_i]))) #DEBUG

    def finalize_chronicler(self):
        self.trial_data['ADSM_roots'] = []
        ADSM_root_ids = [self.Unoccupied.id, self.Hinterland.id] # skip these
        global print_collapse_history
        # dump this collapse history
        def display_faction(root_entry,indent):
            root = root_entry[0]
            ADSM_root_ids.append(root.id) # seen this one
            previous_polity = root.previous_polity
            if previous_polity is None:
                previous_polity_id = -1
                ratio = 0
            else:
                previous_polity_id = previous_polity.id
                # Amount of intersection of max territories_i of parent with child
                ratio = np.double(len(intersect(root.max_territories_i,previous_polity.max_territories_i)))/np.double(len(previous_polity.max_territories_i))
            #DEAD years = self.this_year - root.start_year if root.state is POLITY_ALIVE else root.end_year - root.start_year
            years = root.end_year - root.start_year
            # TODO dump total_population
            self.trial_data['ADSM_roots'].append([root.id,previous_polity_id,root.start_year,
                                                         root.Clon,root.Clat,
                                                         root.starting_size,root.max_size,root.max_year,ratio,years,root.max_population,root.s,root.polity_type,root.continuations,root.state])
            if print_collapse_history: # Don't bother printing since we look at the data via matlab anyway
                print("%s%d %d %d %s %s %d %d %d %.3f %d %.1f %.3f %d" % (indent,root.id,previous_polity_id,root.start_year,
                                                                          root.Clon,root.Clat,
                                                                          root.starting_size,root.max_size,root.max_year,ratio,years,root.max_population,root.s,root.polity_type))
            indent += "   "
            for faction_entry in root_entry[1]: # display the factions root collapsed into
                display_faction(faction_entry,indent)

        # Root polities, by insertion order, are in birth other
        if print_collapse_history:
            print("Collapse history (%d roots):" % len(self.collapse_roots))
        for root in self.collapse_roots:
            if print_collapse_history:
                print("") # separate roots
            display_faction(self.collapse_history[root],"")
        for polity in self.polities:
            # Critical to use large_enough() since NTs are filtered using large_enough_ignore.
            if polity.id not in ADSM_root_ids and polity.large_enough(dead_or_alive=True):
                polity.previous_polity = None # supress ratio
                display_faction([polity,[]],"")
        # TODO dump ALL polities except Hinterland and Unoccupied with stats
        # then roots data need only report ids and ratio
        global report_annexations
        if report_annexations:
            self.trial_data['annexation_data'] = self.annexation_data
        global report_polity_history
        if report_polity_history:
            self.trial_data['polity_history']  = self.polity_history
            self.trial_data['collapse_data']   = self.collapse_data

        super().finalize_chronicler()

if __name__ == "__main__":
    execute(NADSM)
