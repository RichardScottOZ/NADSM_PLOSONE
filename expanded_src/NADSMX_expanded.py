# For use with driver.py
from NADSM import *
import math
import sys
import os
import os.path
import Utils
import numpy as np

Pontic_Caspian_lon = 45.0 # between 40 and 50 really
China_steppe_border_lon = 112.0
horse_cavalry_diffusion_distance = (China_steppe_border_lon - Pontic_Caspian_lon)*111.32*np.cos(np.radians(40))

class NADSMX(NADSM):
##f NADSM:__init__(...)
##f WHM:advance_history(...)
##f NADSM:bounded_s(...)
##f NADSM:chronicle(...)
##f NADSM:compute_min_steppe_distance(...)
##f NADSM:compute_tribes(...)
##f PopChronicler:compute_world_population(...)
##f PopChronicler:compute_world_population_OLD(...)
##f NADSM:deus_ex_machina(...)
##f NADSM:establish_territories(...)
##f NADSM:finalize_chronicler(...)
##f ComparativeHx:finalize_display(...)
##f ComparativeHx:initialize_chronicler(...)
##f ComparativeHx:initialize_display(...)
##f WHM:load_geography(...)
##f NADSM:possibly_create_new_states(...)
##f NADSM:possibly_invent_and_diffuse_nomadic_horse_cavalry(...)
##f NADSM:possibly_update_long_distance_efficiencies(...)
##f Population:set_intensification(...)
##f NADSM:set_population(...)
##f Chronicler:set_pretty_years(...)
##f NADSM:setup_chronicler_for_trial(...)
##f NADSM:setup_for_trial(...)
##f NADSM:setup_nomads_for_trial(...)
##f NADSM:setup_regions_for_trial(...)
##f NADSM:state_of_the_world(...)
##f WHM:unfold(...)
##f NADSM:update_chronicle(...)
##f PopChronicler:update_display(...)
##f NADSM:update_display_figure_hook(...)
##f ComparativeHx:update_display_for_trial(...)
##f Population:update_intensification(...)
##f PopElites:update_population(...)
##f NADSM:update_power_projection(...)
##f WHM:verify_integrity(...)

    def initialize_parameters(self):
##s NADSM.py:NADSM:initialize_parameters
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
##< NADSM.py:NADSM:initialize_parameters


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
##e NADSM.py:NADSM:initialize_parameters
        
        # Override the normal parameters for ALL trials here
        # If you want to change a parameter according to trial do it in setup_for_trial() below
        try: # in a try block in case a self.trial access fails
                
            # Experiments where and when horse cavalry is invented
            if True:
                # play with the default nomadic diffusion times
                cavalry_start_year = -1000 # year when first steppe nomad territory perfects mounted warfare
                cavarly_china_year =  -400 # Year when mounted warfare arrives on NW China border
                cavalry_invention = (Pontic_Caspian_lon,Pontic_Caspian_lon,cavalry_start_year,None) # stock (start spreading in PC in 1000BCE)
                # cavarly_china_year = -200 # A test to see how the other statistics change of NC in the east is delayed
                # cavarly_china_year = -700 # A test to see how the other statistics change of NC in the east is sped up (ala, effectively, ADSM)
                # DEAD self.mounted_warfare_diffusion_start_year = cavalry_start_year
                if 'HC_3M_anywhere' in self.options.X:
                    cavalry_invention = (None,None,None,3) # Invent wherever and whenever an NT can extort from 1M people
                if 'HC_China_1000BCE' in self.options.X:
                    cavalry_invention = (China_steppe_border_lon,China_steppe_border_lon,cavalry_start_year,None) # Start in China

                self.mounted_warfare_diffusion_time = (cavarly_china_year - cavalry_start_year)
                self.mounted_warfare_invention = cavalry_invention
                


            # Not computing sea and desert battles really speed things up, of course
            if True:
                # trigger population depends sensitively on when camels are finally used for desert travel above
                # the camel is put to work militarily in 900BCE.
                # https://en.wikipedia.org/wiki/Camel#Military_uses
                desert_year_trigger = (-900,6.3) # 900BCE inventing 100 years after nomads; stock
                if 'camels_600BCE_4M' in self.options.X:
                    desert_year_trigger = (-600,4.2) 
                if 'camels_600BCE_6M' in self.options.X:
                    desert_year_trigger = (-600,6.3) 
                
                # NOTE: there has to be a 100km change for it to impact anything!  See NADSM.py
                self.desert_distance_schedule = {-1500: (200,0.5),
                                                 desert_year_trigger[0]:  (400,0.3)} # bigger reach during rise of Persia?
                self.confed_trigger_population = desert_year_trigger[1]                 # self.desert_distance_schedule = {-1500: (400,1.0)} # enable but don't make it easy to travel
                print('Experiment: Camels in %s; NC trigger: %.1fM people' % (pretty_year(desert_year_trigger[0]),
                                                                              self.confed_trigger_population))

            if True:
                # Slowly expand the military range of sea-borne attacks
                # NOTE: there has to be a 100km change for it to impact anything!  See ADSM.py
                geo = self.geography
                # uniformly good, just further each 500yrs
                self.sea_distance_schedule = {-1500: (1*geo.interregion_distance_km, .3),
                                              -1000: (2*geo.interregion_distance_km, .3),
                                               -500: (3*geo.interregion_distance_km, .3)}

            # Various alternate and NULL models that still have NCs and NTs but little or no threat to agrarian states
            # In these cases NCs can still form in the presence of wealth so we still try to predict them
            # Thus it is unlike ADSM in that fashion and more apples to apples

            if 'no_nomads_no_nc' in self.options.X:
                print('Experiment: No nomadic threat ever: s never changes (no Kremer or nomadic shock)')
                self.disable_nomads = True
                self.disable_nc_formation = True
                
            # enable Kremer approximation (in addition to nomads)
            # ds/dt = gsP (Kremer, 1993: Equation 3) where 'A' is taken completely as 's'
            # See kremer_s2.m
            k=2.66011e-12 # military inventions per person per 2.0 years from 1.1 to 2.1 in 3000 years
            if 'Kremer_nomads_nc' in self.options.X: # Disable nomadic threat so only Kremer improvement to agrarian military
                self.s_kremer = k # enable mechanism
                print('Experiment: Kremer enabled, with nomadic pressure: %g inventions/person/time step' % self.s_kremer)

            if 'Kremer_no_nomads_nc' in self.options.X: # Disable nomadic threat so only Kremer improvement to agrarian military
                self.s_kremer = k # enable mechanism
                # open code disabling nomads because we need to permit high eventual s
                # disable nomad s threat when extorting
                # self.disable_nomads = True
                self.nomadic_s_threshold = 1 # ag s never gets close and we never bump
                self.nomadic_s_increase = 0 # how much to increase effective s of a NC when an ag gets close
                # even if we engaged nomadic s would stay small
                self.max_tribal_s = initial_ag_state_s # keep their effective s very low
                # Ensure we grow to max ag s (as g assumes it)
                self.max_polity_s = maximum_ag_state_s
                self.permit_agrarian_confederation = 0 # Ensure no agrarian confederations form here
                # self.disable_nc_formation = True
                if True: # self.disable_nc_formation:
                    self.confed_trigger_population = 500 # 500M trigger disables NC formation
                print('Experiment: Kremer only, no nomadic pressure: %g inventions/person/time step' % self.s_kremer)

            initial_state_sizes = np.nan # make originally big
            if 'early_Shang' in self.options.X:
                self.create_states_instructions = {-1500: [[('EgyptNew', -1500), initial_state_sizes, self.PolityType], # Nile Valley large extent
                                                           [('Neo-Assyria', np.nan), initial_state_sizes, self.PolityType], # Tigris/Ephrates
                                                           [('Hittite', -1500), initial_state_sizes, self.PolityType],],
                                                   -1500: [[('Shang', -1500), initial_state_sizes, self.PolityType],], # MYRV
                                                   }
            if 'Mekong_GV' in self.options.X:
                self.create_states_instructions = {-1500: [[('Gupta', 600), np.nan, self.PolityType],  # Upper Ganges
                                                           [('Chen-La', 700), np.nan, self.PolityType], # Lower Mekong
                                                           ]}

            # Centralize the lore of turning off the nomads
            if self.disable_nomads:
                # disable nomad s threat when extorting
                self.nomadic_s_threshold = 1 # ag s never gets close and we never bump
                self.nomadic_s_increase = 0 # how much to increase effective s of a NC when an ag gets close
                # even if we engaged nomadic s would stay small
                self.max_tribal_s = initial_ag_state_s # keep their effective s very low
                self.max_polity_s = initial_ag_state_s
                self.permit_agrarian_confederation = 0 # Ensure no agrarian confederations form here
                if self.disable_nc_formation:
                    self.confed_trigger_population = 500 # 500M trigger disables NC formation in our chosen period

            name = self.name
            hill_climb_file = '%s_fixed.py' % name
            if os.path.exists(hill_climb_file):
                print('Applying %s...' % hill_climb_file)
                exec(compile(open(hill_climb_file, "rb").read(), hill_climb_file, 'exec'))
            hill_climb_file = '%s_vary.py' % name
            if os.path.exists(hill_climb_file):
                print('Applying %s...' % hill_climb_file)
                exec(compile(open(hill_climb_file, "rb").read(), hill_climb_file, 'exec'))

        except KeyError:
            raise RuntimeError("Trial %d: Unable to determine some NADSMX parameter!" % self.trial)

    
if __name__ == "__main__":
    execute(NADSMX) # critical for hill_climb that n_trials is not specified!
