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

    def initialize_parameters(self):
        super().initialize_parameters() # Let NADSM.py set what it wants, then override for -X experiments below
        
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
