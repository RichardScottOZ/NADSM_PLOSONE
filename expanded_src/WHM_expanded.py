# WorldHistoricalModel

# This encodes a theory of how human WorldHistory might have unfolded, how
# people in polities responded to the territories they found themselves in and
# how they interacted with their neighoring polities

# There are expected to be several WHM's (theories) and then many many 'runs'
# (histories) of that model against specific geographies, random seeds, etc.

# Each WHM adds its own figures and overrides behaviors of polities

# WHM creates its Territories initially (can this be reformulated/added during time?)
# and then it generates polities by fiat or model against those territories

# Each instance contains its (partial recording of) chronicles of P/T unfoldings
# against a specific geography using its behavior

# to run n *identical* simulations you need to make a list of the same instance
# (that has a non-None random_seed)
# to run n *new* simulations of the same parms, you need to make a list of n
# copies of a cycle instance and change/clear the random seed
# for cycle in ncycles: whm.parameters.seeed = None, whm.unfold()
import pickle 
from Constants import *
from Utils import *
from Geography import *
from Chronicler import *
import numpy as np
import random # to set the seed only
import time
import copy
import argparse
# This will speed things up but you need to clear the cache whenever (1) the list of territories changes (set_polity)
# or things like sea and desert attack distances change (since their political or physical status might change)
enable_border_caching = True
# Data structure notes:

# This code uses lists for regions in territories rather than sets because we
# use the region lists to index into geography data.

# This code uses sets, rather than lists, to enumerate territories of a polity
# (and polities of the whm).  This is because sets are hashable and so changing
# which polity 'owns' a territory is VERY fast (set_polity).  This is at the
# infrequent expense of determining all a polity's regions, typically during
# display.

# As a rule there shouldn't be any print statements in this code unless it is for debugging
# If possible move all such to the Chronicler

# Gak!!  Bad form and all that....
# We want easy reference to some common instances in other modules
# from WHM import * does not apparently import the variable whm into the global name space of the other module
# and WHM.whm doesn't work (since using from WHM import * doesn't create a separate module name space)
# and we have to be careful about the timing of when the instance is created and assigned, which is NOT at module load time
# Rather than import a function to access we adopt instead a package global 'whm' in each .py file and have the method
# initialize_parameters(), which is called once just after the WHM instance is called to set it locally before calling supers()
whm = None # the instance of the WorldHistoricalModel we are currently unfolding (package global)

class Littoral(object): # a 'territory' that indicates 'ocean' in a Territory's connections
    def __init__(self,id):
        self.id = id
        self.littoral = True

    def __repr__(self):
        return "O(%s)" % self.id
    
class Territory(object): # a collection of Geography 'regions' that Polity instances can occupy
    # These are abstractions of parts of a geography that might reflect natural borders etc
    # but typically reflect 'valued' resources and other economic assets
    # better name later might be Resources of which Territories might be a subtype for ag
    # but it could be a commodity, mine, a port or a bay or a set of mountains, etc

    # NOTE that we can have territories like Asia, Anatolia, Rus, WesternEurope
    # that overlap with smaller territories which might not be used by an actual
    # historical unfolding but could be used by a deus_ex_machina() or evaluation()
    # method to assert things or eval things (which might mean unfolding to
    # shared regions, etc.)
    
    # Don't assume each region is used uniquely amongst territories (a location
    # could be both good ag and a good port)

    # See OccupiedTerritories subclass
    # TODO assert biome=GEO_DESERT elevation=0
    def __init__(self,regions):
        self.regions = unique(regions) # ensure unique elements and make set
        self.littoral = False
        self.set_properties(regions)

    def DEAD__lt__(self,other):
        # Why are we 'sorting' Territories? 
        # typically other is '0' when we want to see if it is ocean in a list of connections
        # but if we get here then self is clearly a Territory and can't be ocean
        # otherwise, for 'sorting' we use id() since it will be canonical within a run
        # TODO if it needs to be canonical between runs, give explicit id, i.e., self.t_id
        return False if isinstance(other,type(int)) else id(self) < id(other)

    def set_properties(self,regions):
        if len(regions):
            global whm
            geo = whm.geography
            # compute centroid of regions
            self.lat = np.mean(geo.lat[regions])
            self.lon = np.mean(geo.lon[regions])
            biomes = geo.Biome[regions]
            self.Biome = GEO_AGRICULTURAL if GEO_AGRICULTURAL in biomes else biomes[0]
            self.Elevation = max(geo.Elevation[regions])
        
    def __repr__(self):
        return "T(%s)" % succinct_elts(self.regions,matlab_offset=0)

    def setup_for_trial(self):
        global whm
        geo = whm.geography
        self.polity = None

    # Each territory is occupied by a single polity and that polity might be a member of a federation of polities
    # Each polity and fedeartion maintains their list of territories (for a federation it is (should be) the
    # union of all member polity's territories). This method maintains those lists and associated metrics properly
    # and should always be used by actions that change territory polity allegence.
    # Set also Confederation.split and Confederation.join, which also maintain these counts on federations
    # The convention is to always use join or split to update on a confederation

    # These might be called directly when joining a confederation, in which case 'polity' will be the confederation
    # These helper routines deliberately do not check about polity.confederation
    def add_to_polity_lists(self,polity):
        global whm
        if polity.state not in [POLITY_ALIVE, POLITY_MEMBER]:
            if polity.permit_resurrection:
                polity.set_state(POLITY_ALIVE) # we're back
            else:
                raise RuntimeError("How can we be assigning a territory to a dead polity %s" % polity)
        polity.set_border_cache() # clear border cache for political and physical on polity
        if use_set_versions:
            polity.territories.add(self)
        else:
            polity.territories.append(self)
        polity.size += 1
        polity.max_size = max(polity.size,polity.max_size)
        t_i = self.t_index
        polity.territories_i.append(t_i)
        if self.agricultural:
            polity.agricultural_i.append(t_i)
        polity.km2_size = np.sum(whm.area_km2[polity.territories_i])

    def remove_from_polity_lists(self,polity):
        global whm
        polity.set_border_cache() # clear border cache for political and physical on polity
        polity.territories.remove(self)
        polity.size -= 1
        t_i = self.t_index
        polity.territories_i.remove(t_i)
        if self.agricultural:
            polity.agricultural_i.remove(t_i) 
        if polity.size == 0:
            polity.set_state(POLITY_ANNEXED if polity.max_size else POLITY_DEAD)
        polity.km2_size = np.sum(whm.area_km2[polity.territories_i])

    # This is typically used for annexation
    def set_polity(self,polity):
        current_polity = self.polity
        if current_polity is not None:
            if current_polity is polity:
                return
            self.remove_from_polity_lists(current_polity)
            if current_polity.confederation is not None:
                self.remove_from_polity_lists(current_polity.confederation)
        self.polity = polity
        self.add_to_polity_lists(polity)
        if polity.quasi_state:
            self.quasi_state = polity # what this territory ought to return to when collapsing
        if polity.confederation is not None:
            self.add_to_polity_lists(polity.confederation)
        #DEBUG whm.verify_integrity() #DEBUG

    def get_connections(self,sea=0,desert=0,combined=False):
        # Return a list of territories (or littoral objects)
        sea_connections = []
        if sea:
            for entry in self.sea_connections:
                if entry[1] <= sea: # within distance
                    sea_connections.append(entry[0])
                else:
                    break # the sea_connections are sorted by distance
        desert_connections = []
        if desert:
            for entry in self.desert_connections:
                if entry[1] <= desert: # within distance
                    desert_connections.append(entry[0])
                else:
                    break # the desert_connections are sorted by distance
        if combined: # want a single list for iteration?
            desert_connections.extend(self.land_connections)
            sea_connections.extend(desert_connections)
            return sea_connections
        else:
            return (self.land_connections,sea_connections,desert_connections)

polity_counter = 1 # for debugging
class State(object):
    def __init__(self,name=None):
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
    def make_quasi_state(self,arising_polity_type=None):
        self.quasi_state = True
        self.large_enough_ignore = True # never passes muster to be used in statistics
        self.permit_resurrection = True # territories come and go
        self.set_state(POLITY_ALIVE) # will never die, collapse, or become a member, etc.
        self.arising_polity_type = arising_polity_type # if not None, allowed to form a polity
        
    def large_enough(self,dead_or_alive=False):
        # for tracking and reporting purposes, is/was this polity big eough to matter?
        global whm
        if self is None or self.large_enough_ignore:
            return False
        elif self.state is POLITY_ALIVE and self.size >= whm.compare_polity_size:
            return True
        # Must have been DEAD or something...look at its biggest size
        elif (dead_or_alive and self.max_size >= whm.compare_polity_size):
            return True
        else:
            return False

    # So subclasses can hook this...
    def set_state(self,state):
        if not self.quasi_state: # We never change the state of quasi_states
            self.state = state

    def same_polity(self,polity):
        # either the same polity instance or self is a member of a confederation and so is polity
        # self.is_confederation and polity.confederation is self
        # polity.is_confederation and self.confederation is polity
        return (self is polity) or (self.confederation and self.confederation is polity.confederation)

    def machiavelli(self):
        # estimate threats and write orders, which are returned
        return []

class Polity(State): # tribes, states

    def __init__(self,name=None,territories=None):
## super calls State:__init__(...)
        super().__init__(name=name)
        territories = [] if territories is None else territories # supply mutable default value
        self.is_confederation = False
        self.confederation = None
        # UNNEEDED since set_polity() will clear cache each time
        # self.set_border_cache() # initialize border cache for political and physical as empty
        for territory in territories:
            territory.set_polity(self)
        
    def __repr__(self):
        return "P(%s[%s])" % (self.name,polity_state_tags[self.state])

    def same_polity(self,polity):
        return (self is polity or
                self.confederation is not None and self.confederation is polity.confederation)

    # which territory of self annexed what territory of some other poor polity
    def annex(self,victorious_territory,vanquished_territory,update_state=True,hinterland_arise=1.0):
        # possibly returns new polity, else self
        global whm
        Hinterland = whm.Hinterland
        # TODO show where battle was fought and exchanged (A2)
        # TODO: track, on vanquished polity, the polity (self) that last annexed something and the year whm.this_year for invasion analysis
        # TODO: in ADSM write a supers that records on vanquished polity if the annexed territory was the capital territory: who and year
        polity = self
        if self.quasi_state and self.arising_polity_type is not None:
            if urandom_random() < hinterland_arise:
                polity = self.arising_polity_type()
                victorious_territory.set_polity(polity)
            else:
                # the vanquished becomes (or remains) Hinterland
                polity = Hinterland 
        vanquished_territory.set_polity(polity)
        if update_state and polity is not Hinterland:
            # update stats and power of victor
            polity.state_of_the_union()
            # BUG what about vanquished_territory polity?
        return polity

    def collapse(self):
        raise RuntimeError("Why is %s collapsing??" % self)
        
    def state_of_the_union(self):
        # only called when POLITY_ALIVE
        global whm
        self.end_year = whm.this_year # CRITICAL update to latest year alive
    

    def set_border_cache(self,political=None,physical=None):
        global enable_border_caching
        if not enable_border_caching and (political or physical):
            return
        # Enabled; will reset when political and physical are None
        self._political_territories = political
        self._physical_territories = physical

    # This is an expensive computation that can happen each time step for each polity
    # First order speed improvement is to cache the result and flush the cache when polity territories change
    # This lead, for ADSM in OldWorld+IV, a 25% improvement in overall speed from 105s to 79s
    # Alternatively we could maintain a list of all border cells (both physical and political)
    # and a list of border changes because of gains and losses
    # if there are no gains or losses, return the lists as is (ala the caching version)
    # if there are gains see if they are borders now and eliminate other gains/original borders now saturated (core)
    # if there are losses see which of their neighbors are still self and add those to borders if needed, then remove losses from borders
    # All this to be gain/loss driven and skip proving core is still the core
    # Initially everything is 'gain' but the core will be proven not a border
    def find_borders(self,sea=0,desert=0,physical=False):
        # which of self's territories shares a border with a different polity? (political)
        # use land_connections only
        # if physical return all the physical borders as well

        # if border cache available use it, else compute
        global whm
        if self._political_territories is not None:
            if physical:
                return (self._political_territories, self._physical_territories)
            else:
                return self._political_territories

        Unoccupied = whm.Unoccupied
        if use_set_versions:
            _init = set  # ensures unique BUT at the expense of replicating order!!
            _add  = set.add
        else:
            _init = list
            _add  = list.append
        political_border_territories = _init()
        physical_border_territories = _init() 
        for territory in self.territories:
            # Determine if this territory is a border of some sort
            is_political = False
            is_physical  = False
            (land_connections,sea_connections,desert_connections) = territory.get_connections(sea=sea,desert=desert,combined=False)
            if len(land_connections) < 4:
                # Under the VN assumption of create_geography,
                # if there are less than 4 connections we are at the land-edge
                # of a continent/arena. Oceans at littoral boundaries
                # are encoded explicitly (PNAS encoding) but we need to see if there are no sea connections.
                is_physical = True

            # before we extend the list, record the number of sea and desert connections
            n_sea_connections = len(sea_connections)
            n_desert_connections = len(desert_connections)
            # Form a single, combined list of connections (what 'combined' would do)
            # reflects knowledge of get_connections(), which creates separate, disposable lists for current sea and desert connections
            # Thus extend() does NOT mess with land_connections list
            desert_connections.extend(land_connections)
            sea_connections.extend(desert_connections)
            # now, for this territory, look at all its connections and determine if it is a physical and/or political bounday
            for connected_territory in sea_connections: # actually all connected regions on our border 
                if connected_territory.littoral: # An ocean?
                    if n_sea_connections == 0:
                        is_physical = True
                        if is_political:
                            break # enough with this territory
                    continue # look at the rest of the connections

                connected_polity = connected_territory.polity
                if not self.same_polity(connected_polity):
                    # in the case of desert we typically have Unoccupied so it isn't political
                    # if there are no desert_connections then it is physical, else it might be political
                    if connected_polity is None or connected_polity is Unoccupied: # really physical
                        if whm.geography.Biome[connected_territory.t_index] is not GEO_DESERT or n_desert_connections == 0:
                            is_physical = True
                            if is_political:
                                break # enough with this territory
                    else:
                        is_political = True
                        if physical:
                            if is_physical:
                                break # enough with this territory
                        else:
                            break # enough with this territory
            # record territory types if applicable
            if is_political:
                _add(political_border_territories,territory)
            if physical and is_physical:
                # Add discount factor
                _add(physical_border_territories,territory)

        if not use_set_versions:
            # ensure these are unique via our version in Utils that handles lists and preserves order
            political_border_territories = unique(political_border_territories)
            physical_border_territories  = unique(physical_border_territories)

        # update border caches:
        self.set_border_cache(political_border_territories,physical_border_territories)
        if physical:
            return (political_border_territories,physical_border_territories)
        else:
            return political_border_territories

    def find_annexable(self,sea=0,desert=0,po_t=None):
        # find political borders and then return a list of territories of those borders
        # that belong to someone else
        global whm
        Unoccupied = whm.Unoccupied
        if po_t is None:
            po_t = self.find_borders(sea=sea,desert=desert,physical=False);
        annexable_territories = []
        for p_t in po_t:
            all_connections = p_t.get_connections(sea=sea,desert=desert,combined=True)
            for connected_territory in all_connections:
                if connected_territory.littoral:
                    continue # skip ocean markers

                connected_polity = connected_territory.polity
                if not self.same_polity(connected_polity):
                    if connected_polity is None or connected_polity is Unoccupied: # really physical
                        continue # not annexable
                    else:
                        annexable_territories.append(connected_territory)
        return unique(annexable_territories)

class Confederation(State):
    def __init__(self,name=None):
## super calls State:__init__(...)
        super().__init__(name=name)
        self.is_confederation = True
        self.core_polity = None
        self.member_polities = [] # join and split manage this list
        
    def assert_core_polities(self,core_polity=None,polities=None):
        if core_polity:
            # have the core
            self.core_polity = core_polity
            self.join(core_polity)
        # and perhaps the rest of the polities join the federation at this time
        polities = [] if polities is None else polities # supply mutable default value
        for polity in polities:
            self.join(polity)

    def __repr__(self):
        state = {POLITY_ALIVE:     'A',
                 POLITY_DEAD:      'D',
                 POLITY_ANNEXED:   'X',
                 POLITY_COLLAPSED: 'C',
                 }[self.state]
        return "C(%s[%s])" % (self.name,state)

    def same_polity(self,polity):
        # self is a confederation, poloty might be or might be a member
        return super().same_polity(polity) or polity.confederation is self

    def collapse(self):
        self.core_polity = None # clean up and avoid split() recursion/error
        for polity in copy.copy(self.member_polities):
            self.split(polity)
        self.set_state(POLITY_COLLAPSED)

    def state_of_the_union(self):
        global whm
        if self.state is not POLITY_ALIVE:
            # constraint on state_of_the_union()
            raise RuntimeError("state_of_the_union() called for %s but not alive?!?" % self)
            
        for polity in self.member_polities:
            polity.state_of_the_union() # This is the one place where sou() gets called with POLITY_MEMBER
        self.end_year = whm.this_year # CRITICAL update to latest year alive
    
    def join(self,polity):
        if polity in self.member_polities:
            raise RuntimeError("Polity %s already a member of %s?!?" % (polity,self))
        
        polity.confederation = self
        polity.set_state(POLITY_MEMBER)
        self.member_polities.append(polity)
        # The territories are still owned by polity so no need to use set_polity
        for territory in polity.territories:
            territory.add_to_polity_lists(self)
        polity.set_border_cache() # must reset since neighbors might now be friends

    def split(self,polity):
        # explicity remove territories from confedeartion's list and adjust size
        # The territories are still owned by polity so no need to use set_polity
        if self.core_polity == polity:
            # raise RuntimeError, "Core polity %s splitting from %s! COLLAPSE?" % (polity,self)
            self.collapse()

        polity.confederation = None
        self.member_polities.remove(polity)
        for territory in polity.territories:
            territory.remove_from_polity_lists(self)
        polity.set_border_cache() # must reset since neighbors might now be enemies
        polity.set_state(POLITY_ALIVE)

# ExecutiveOrder
# Can you have standing orders that don't get reset each year?
# Lost only when explicitly reset or when polity collpases/annexed
# These are laws of the land...

# Each polity has its list of orders and contributes them to the world
# NOTE: these objects are tagged as Reusable, which means they are recycled
# rather than GC'd.  Is this needed?
# In any case, ensure any __init__() methods clear/reset needed instance variables
class Order(object): # Don't use Reusable py3 has issues
    def execute():
        pass

# War on a front, a type of Order
class Front(Order):
    # us_territory
    # us_polity
    # them_territory
    # them_polity

    def win():
        return us_power >= them_power

    def war():
        if us_territory.polity == us_polity and them_territory.polity == them_polity:
            # This front order is still valid
            # if we are being offensive, and we seem to have won
            if us_state == OFFENSIVE and self.win():
                victorious = us_polity
                vanquished = them_polity
                if victorious.quasi_states and victorious.arising_polity_type is not None:
                    # promote to a new polity and add to History
                    # need a factory class instance
                    pass


## defined last so we can reference Polity and Territory
class WorldHistoricalModel(Chronicler, object):
##f Chronicler:chronicle(...)
##f Chronicler:finalize_chronicler(...)
##f Chronicler:finalize_display(...)
##f Chronicler:initialize_chronicler(...)
##f Chronicler:initialize_display(...)
##f Chronicler:set_pretty_years(...)
##f Chronicler:setup_chronicler_for_trial(...)
##f Chronicler:update_chronicle(...)
##f Chronicler:update_display(...)
##f Chronicler:update_display_figure_hook(...)
##f Chronicler:update_display_for_trial(...)
    # how mankind's polities unfold in time

    # Initial thoughts about instance variables
    # type_name # a class pretty text name for the model type, e.g., Historical PNAS (add /pop) ADSM 
    # name # a specific title for this instance (date created, fragments of parameters, etc.)
    # TODO parameters <instance of an associated parameter structure>
    # Chronicles[] # partial deepcopy instances of the WHM at certain times
    

    # These are constantly in flux for the current year
	# this_year: <current year>
    # territories[]
    # polities[]
    # orders[]

    # These class variables are used to create the proper associated types of Polities and Territories for this WHM
    # Each specialization of WHM overrides them ala a 'mixin' with the one it wants for any quasi_states
    # TODO rename these variables to make that explicit and have the model specialization be direct with the polity
    # types it wants to create when...  TerritoryType is similar.
    # TODO permit PolityType as none, in which case no Hinterland or Unoccupied polity get created?  Is that a good thing?
    PolityType = Polity # class variables for the polity 'factory'
    TerritoryType = Territory # class variable for the territory 'factory'

    def __init__(self,options):
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

    def load_geography(self):
        geography_name = self.geography_name
        filename = os.path.join(os.path.abspath(self.options.g),"%s.pkl" % ensure_basename(geography_name))
        try:
            fh = open(filename,"rb")
            print("Loading %s" % geography_name) # DEBUG
            if sys.version_info[:3] >= (3,0):
                self.geography = pickle.load(fh,encoding='latin1') # built with numpy arrays
            else:
                self.geography = pickle.load(fh)
            fh.close()
        except:
            raise RuntimeError("Unable to load %s!" % filename)
        geo = self.geography
        region_scale = 1 # 1 region per territory
        if geo.actual_lat_lon:
            self.area_km2 = geo.equatorial_region_km2*region_scale*np.cos(np.radians(geo.lat))
        else:
            # PNAS (and A2, ec.) had no cos(lat) dependence and used this rough size per territory
            self.area_km2 = geo.equatorial_region_km2*region_scale*np.ones(geo.n_regions,dtype=float )

    def initialize_parameters(self):
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

    def establish_territories(self):
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

    def setup_for_trial(self):
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
    def unfold(self,trial):
        # move call to initialize_chronicler here?
        # perhaps reset_chronicler?
        self.start_time = time.time() # restart timer
        self.trial = trial
        try:
            self.setup_for_trial()
            print_scalar_members(self,header="Parameters for trial %d:" % trial,footer="",indent="   ") # DEBUG

            for self.this_year in self.years:
                self.this_pretty_year = pretty_year(self.this_year) # make available
                self.advance_history()

            print('End of history') # BREAK here to inspect the final datastructures
            print('Unfold time (%d): %.1fs' % (trial,time.time() - self.start_time))
            self.finalize_chronicler()
        except RuntimeError as exception:
            import traceback
            print("Unexpected error:", traceback.format_exc())

    # This is the main method that moves things along each 'year'
    def advance_history(self):
        # Advance history
        self.orders = [] # reset here in case deus_ex_machina() wants to use this mechanism for its fiat

        self.deus_ex_machina() # possibly update the world and people by fiat
        # Fait accompli...

        # If we collapse during state_of_the_union() or create polities during war
        # self.polities strictly grows and we will enumerate them here...
        for polity in self.polities:
            if polity.state is POLITY_ALIVE:
                # update state, determine dead, current borders, bordering polities
                polity.state_of_the_union() 
        # perform world-wide calculations for all polities.
        self.state_of_the_world()

        # The world is somewhat stable so checkpoint.
        self.chronicle()
        
        for polity in self.polities:
            if polity.state is POLITY_ALIVE:
                self.orders.extend(polity.machiavelli())

        if False and self.orders: # DEBUG
            print('%s: %d orders' % (self.this_pretty_year,len(self.orders)))
        urandom_shuffle(self.orders)
        #DEBUG print self.orders #DEBUG
        for order in self.orders:
            order.execute()
        
    def deus_ex_machina(self):
        # Acts of 'god': possibly over multiple territories in this year;
        # Most of the time this is a pass.
        pass

    def state_of_the_world(self):
        pass

    def verify_integrity(self,verbose=False):
        # test that the various main datastructures are following the rules
        # be careful where you call this since there are moments as in set_polity
        # when the datastructures can be tempporarily out of sync
        if verbose:
            print('Verifying: %d total polities' % len(self.polities))
        # map over each polity and verify that each listed territory belongs to it
        for polity in self.polities:
            if polity.state is POLITY_ALIVE:
                # check that there ARE territories
                for territory in polity.territories:
                    if territory.polity is not polity:
                        print('%s should be %s but is not: %s' % (territory,polity,territory.polity))
            else:
                # check that self.territories are empty
                pass
        # A second test would be to map over self.territories and ensure that
        # if territory.polity is not None that it is in the list of the polity's territories

if __name__ == "__main__":
    # Yes, this structual placeholder set of class executes...and does nothing...
    # It creates 1:1 territories from a geography but creates no polities
    execute(WorldHistoricalModel)
