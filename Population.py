from WHM import *
whm = None # package global
from Utils import *
from Geography import *
from PopChronicler import *
import numpy as np

# For speed, we implement population and k as numpy arrays over the entire set of territories
# for rapid update and each polity maintains a mask into these arrays.

class OccupiedTerritory(Territory):
    # init: set initial population according to Biome:
    def set_properties(self,regions):
        super().set_properties(regions)
        if len(regions):
            self.agricultural = (self.Biome == GEO_AGRICULTURAL) # cache this

    def setup_for_trial(self):
        super().setup_for_trial()
        # Initialize territorial parameters and starting population, birth
        # rates, and expected k for state or hinterland
        global whm
        t_i = self.t_index
        # setup initial carrying capacity, which is always unoccupied_density until assigned by a polity
        whm.k[t_i] = whm.unoccupied_density

    def set_polity(self,polity):
        # Before: move the territory index before we lose track of who owns it now
        global whm
        previous_polity = self.polity
        # at the beginning there is no polity to update
        if previous_polity is None:
            previous_polity = whm.Unoccupied # None is like Unoccupied

        # If a territory was previously Unoccupied initialize its population
        # after this, we let history take its course
        t_i = self.t_index
        if (previous_polity is whm.Unoccupied and polity is not whm.Unoccupied and # transitioning
            self.agricultural and whm.population[t_i] == 0): # no population yet...
            self.initialize_population(whm,polity,t_i)
        super().set_polity(polity)
        
    def initialize_population(self,whm,polity,t_i):
        # initialize population in a territory
        initial_population = whm.initial_population_density*whm.intensification_factor[t_i]*whm.area_km2[t_i]
        whm.population[t_i] = initial_population

class DemographicPolity(Polity):

    def __init__(self,name=None,territories=None):
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

        super().__init__(name=name,territories=territories)

    def set_state(self,state):
        super().set_state(state)
        if self.state not in [POLITY_ALIVE, POLITY_MEMBER]:
            # Ensure that DEAD polities have no (total_)population
            # or you will get bad world_population counts
            self.total_population = 0
            # do not reset max_population
            self.total_k = 1 # POLITY_DEAD but avoid DBZ in logistic calculation below
            
    
    def state_of_the_union(self):
        super().state_of_the_union()

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
    def update_arable_k(self):
        global whm
        t_i = self.agricultural_i # only over the agricultural regions
        whm.k[t_i] = self.density*whm.area_km2[t_i]*whm.intensification_factor[t_i]
        whm.birth_rate[t_i] = self.br

    # function that takes migration_factor, array of current population and k and returns motion
    def compute_migration(self,migration_factor,population,k):
        # self.total_population might be incorrect if wars have happened
        n_t = len(population)
        migration = np.zeros(n_t,dtype=float ) # no one moves
        total_population = np.sum(population)
        if total_population: # only move if there are people
            total_k = np.sum(k)
            equal_k = k/total_k # equally distributed total opportunity (not residual) given regional variation
            equal_p = total_population*equal_k # current population distributed according to equal opportunity, if all could move
            local_motion_needed = equal_p - population # comings and goings required for equality
            # Constraint np.sum(local_motion_needed) == 0
            # We have regions that have a surplus and others that have a deficit for equal distribution of people
            # It should be that, if the constraint is true above, that the sum of the surplus is the sum of the deficits
            # so that sum of the abs() of both of those with be 2x the total number of people available to move if migraion_fraction=1.0
            # that is, the comings and goings should match and hence the total motion is 1/2 of the coming and goings
            total_motion_needed = np.sum(np.abs(local_motion_needed))/2
            # shouldn't the movable_feast be migraion_fraction*total_motion_needed (fraction of total amount to equalize)
            if total_motion_needed > 1: # at least 1 person needs to move?
                movable_feast = migration_factor*total_motion_needed # how many needed people are permitted to move
                migration = movable_feast*(local_motion_needed/total_motion_needed) # move proportionally to need
        return migration # Constraint: sum(migration) == 0
        
    def migrate_population(self,migration_factor=0):
        if migration_factor:
            # spatial migration of population within-polity
            # determine available opportunities and move population around toward equitable distribution slowly
            global whm
            t_i = self.agricultural_i # only over the agricultural_i
            migration = self.compute_migration(migration_factor,whm.population[t_i],whm.k[t_i])
            whm.population[t_i] += migration
            

class Population(PopChronicler,WorldHistoricalModel):
    PolityType = DemographicPolity
    TerritoryType = OccupiedTerritory

    def initialize_parameters(self):
        global whm
        whm = self # set package global
        super().initialize_parameters()
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

    def establish_territories(self):
        super().establish_territories()
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
    
    def update_intensification(self,year,announce=True,adjust_hinterland=True):
        if adjust_hinterland:
            try:
                hinterland_factor = self.hinterland_density_factor[year]
            except KeyError:
                pass
            else:
                self.Hinterland.density = self.hinterland_density*hinterland_factor
                if announce:
                    print("%s: Changing hinterland density to %.2f" % (self.this_pretty_year,self.Hinterland.density))

        if self.use_regional_k:
            try:
                instructions = self.regional_intensifications[year]
            except KeyError:
                pass # nothing to do
            else:
                for region,factor in instructions:
                    regions_i = self.region_territories_i[region]
                    n_total_regions = len(regions_i)
                    regions_i = [reg_i for reg_i in regions_i if self.intensification_factor[reg_i] != factor]
                    if len(regions_i):
                        if announce:
                            print("%s: Changing intensification in %s(%d/%d) to %.2f" % (self.this_pretty_year,region,len(regions_i),n_total_regions,factor))
                        self.intensification_factor[regions_i]  = factor

    def set_intensification(self,factor,regions_i):
        self.intensification_factor[regions_i] = factor

    def setup_for_trial(self):
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
        super().setup_for_trial()


        # CRITICAL *reset* k and br once for these special polities
        self.Unoccupied.density = self.unoccupied_density
        self.Unoccupied.br = 0 # no growth with this 'polity'
        self.Hinterland.density = self.hinterland_density
        self.Hinterland.br = self.hinterland_birth_rate

    def deus_ex_machina(self):
        super().deus_ex_machina()
        
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

    def state_of_the_world(self):
        super().state_of_the_world()
        self.update_population() # PopElite overrides

    def set_population(self,polity,population_density,territories_i):
        self.population[territories_i] = population_density*self.intensification_factor[territories_i]*self.area_km2[territories_i] # ADSM eqn 2
        
    def update_population(self):
        # update the population after all the wars
        # do this enmass since it is independent of polity
        # GROW logistically
        current_population = self.population
        opportunity = (1 - current_population/self.k)
        population_change = self.birth_rate*opportunity*current_population # ADSM eqn 1
        if self.k_collapse_fraction:
            # if there has been an exogenous sharp drop in k locally
            # opportunity will be negative (and population change computed above will be negative)
            # Here we model death as an immediate die-off to some fraction of the available k
            k_collapse_i = np.nonzero(opportunity < 0)[0]
            # Compute the die off. The change is the difference between current population and delta*K
            population_change[k_collapse_i] = -(current_population[k_collapse_i] - self.k[k_collapse_i]*self.k_collapse_fraction) # ADSM eqn 3
            # accumulate the die off here for reporting?
            # self.k_collapse_deaths += np.sum(population_change[k_collapse_i])
            
        # make sure we never kill off more people than we have in any one region
        # NOTE: this code assumes current_population is positive or zero
        # hence the negative of it is the most we can lose
        population_change = np.maximum(population_change,-current_population)
        self.population += population_change
            
