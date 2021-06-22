# like Population but tracks elite and peasant population sectors separately
# elite is assumed to be a fraction of total
# migration can be spatial like Population but could be between classes
from WHM import *
whm = None # package global
from Utils import *
from Population import *
import numpy as np
# Ensure that DEAD polities have no (total_)population
# or you will get bad world_population counts
class EliteOccupiedTerritory(OccupiedTerritory):

    def initialize_population(self,whm,polity,t_i):
        # initialize population in a territory based on its new polity
        super().initialize_population(whm,polity,t_i)
        initial_population = whm.population[t_i] # whatever the main calc computed
        # split it as though br was identical and k pie was divided equally since 'the beginning'
        # NOTE: see note below about accumulating differences between population and its components!
        # it starts right here...
        whm.elites[t_i]   = initial_population*polity.elite_k_fraction
        whm.peasants[t_i] = initial_population - whm.elites[t_i]

class EliteDemographicPolity(DemographicPolity):

    def __init__(self,name=None,territories=None):
        territories = [] if territories is None else territories
        super().__init__(name=name,territories=territories)
        global whm
        self.elite_k_fraction = whm.base_elite_k_fraction

    # This should be called whenever the set of territories changes
    def update_arable_k(self):
        super().update_arable_k()
        global whm
        t_i = self.agricultural_i # only over the agricultural regions
        elite_k_fraction = self.elite_k_fraction
        whm.elite_k_fraction[t_i] = elite_k_fraction # propagate to territories in case it changed

    def migrate_population(self,migration_factor=0):
        # This method completely overrides the one in Population
        # Ignore the passed value of migration factor
        # spatial migration of population
        # determine available opportunities and move population around toward mean slowly
        global whm
        t_i = self.agricultural_i # only over the agricultural_i
        # In this version we migrate elites and peasants separately
        # so no promotion/demotion between classes
        if whm.migration_fraction_elites:
            elite_migration = self.compute_migration(whm.migration_fraction_elites,whm.elites[t_i],whm.k[t_i]*self.elite_k_fraction)
            whm.elites[t_i] += elite_migration
        if whm.migration_fraction_peasants:
            peasant_migration = self.compute_migration(whm.migration_fraction_peasants,whm.peasants[t_i],whm.k[t_i]*(1 - self.elite_k_fraction))
            whm.peasants[t_i] += peasant_migration
        # Finally update total population in each territory to reflect the results of the moves, if any
        whm.population[t_i] = whm.peasants[t_i] + whm.elites[t_i]


class ElitePopulation(Population):
    PolityType = EliteDemographicPolity
    TerritoryType = EliteOccupiedTerritory

    def initialize_parameters(self):
        global whm
        whm = self # set package global
        super().initialize_parameters()

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


    def setup_for_trial(self):
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

        super().setup_for_trial()

    def set_population(self,polity,population_density,territories_i):
        super().set_population(polity,population_density,territories_i)
        # apportion what the super() set between elite and peasants
        pop = self.population[territories_i]
        elite_fraction = self.base_elite_k_fraction
        peasant_fraction = 1.0 - elite_fraction
        self.elites  [territories_i] = pop*elite_fraction
        self.peasants[territories_i] = pop*peasant_fraction

    def update_population(self):
        # update the population after all the wars
        # do this enmass since it is independent of polity
        # GROW each cohort logistically

        # Explicitly NO call to super() so we don't double kill the population

        # NOTE: These are the opportunities presently, not after growing, so there is a slight lag
        current_population = self.elites
        elite_k = self.k*self.elite_k_fraction
        elite_opportunity = (1 - current_population/elite_k) # ADSM eqn 5 opportunity
        elite_population_change = self.birth_rate*elite_opportunity*current_population # ADSM eqn 5
        if self.k_collapse_fraction:
            # if there has been an exogenous sharp drop in k locally
            # opportunity will be (substantially) negative
            k_collapse_i = np.nonzero(elite_opportunity < 0)[0]
            # compute the die off
            # This replaces the (too small) negative values already in these locations
            # Unless collapse_faction is larger than 1 (which will grow people) or less than 0 this will never exceed the population available
            elite_population_change[k_collapse_i] = elite_k[k_collapse_i]*self.k_collapse_fraction - current_population[k_collapse_i]
            # accumulate the die off here for reporting?
            # self.k_collapse_deaths += np.sum(elite_population_change[k_collapse_i])

        current_population = self.peasants
        peasant_k = self.k*(1 - self.elite_k_fraction)
        peasant_opportunity = (1 - current_population/peasant_k) # ADSM eqn 4 opportunity
        peasant_population_change = self.birth_rate*peasant_opportunity*current_population # ADSM eqn 4

        if self.k_collapse_fraction:
            # if there has been an exogenous sharp drop in k locally (transition from state to hinterland, famine, etc.)
            # opportunity will be (substantially) negative
            k_collapse_i = np.nonzero(peasant_opportunity < 0)[0]
            # compute the die off
            # This replaces the (too small) negative values already in these locations
            # Unless collapse_faction is larger than 1 (which will grow people) or less than 0 this will never exceed the population available
            peasant_population_change[k_collapse_i] = peasant_k[k_collapse_i]*self.k_collapse_fraction - current_population[k_collapse_i]
            # accumulate the die off here for reporting?
            # self.k_collapse_deaths += np.sum(peasant_population_change[k_collapse_i])

        # TODO promote/demote population between cohort classes here

        # update the globals

        self.elites   += elite_population_change
        self.elites[np.nonzero(self.elites <0)[0]] = 0 # ensure no negative populations
        self.peasants += peasant_population_change
        self.peasants[np.nonzero(self.peasants <0)[0]] = 0 # ensure no negative populations
        self.population = self.elites + self.peasants
        # recompute these rather than using cached versions since we might have killed off a lot of people in a k collapse
        self.elite_opportunity   = (1 - self.elites/elite_k)
        self.peasant_opportunity = (1 - self.peasants/peasant_k)
            
