# python NADSMX_driver.py &

# Run an experiment but tag the directory with a different name (typically you'd specify -X at the end)
#  python3 NADSMX_driver.py -z --n_trials 32 -t stock &
#  python3 NADSMX_driver.py -z --n_trials 32 -- -X camels_600BCE &

# Run a generator:
#  python3 NADSMX_driver.py  -g elite_misery -z --n_trials 32 &
#  python3 NADSMX_driver.py  -g elite_misery -z --n_trials 32 -- -X early_Shang &

# to kill an errant job:
# pkill NADSMX_driver
# kill $(ps aux | grep '[pP]ython NADSMX_driver' | awk '{print $2}')
# then wait for the system'd subprocesses in background to die as well
from Utils import * # driver()
if __name__ == "__main__":
    # For sea/desert scale experiments
    sea_desert_consts = ['rgn_dist = self.geography.interregion_distance_km',
                         'sea_distance_schedule    = {-1500: (1*self.rgn_dist, self.sea_eff),-1000: (2*self.rgn_dist, self.sea_eff),-500: (3*self.rgn_dist, self.sea_eff)}',
                         # 'sea_distance_schedule    = {-1500: (3*self.geo.interregion_distance_km, self.sea_eff)}',
                         # Ensure -900 for camels and middling efficiency for mules/donkeys before that
                         # 'desert_distance_schedule = {-1500: (200, self.desert_eff),-900: (400, self.desert_eff)}',
                         'desert_distance_schedule = {-1500: (200, 0.5),-900: (400, self.desert_eff)}',
                         ]
    geographies = ["OldWorld_HF50"] # 50% replacement of steppes in Hilly Flanks and Persian region
    driver("NADSMX", # code to run
           ["NADSM.py","PopElites.py"], # implementation code
           geographies,
           generators={
                       'attack_budget':   {'attack_budget_fraction': [1/8.0, 1/4.0, 1/3.0, 1/2.0, 1/1.5, 1/1.25, 1/1.0]}, # fraction of political border per time step
                       'elite_fraction': {'base_elite_k_fraction': [0.01, 0.05, 0.10, 0.15, 0.20]}, # epsilon
                       'elite_migration': {'migration_fraction_elites': [0, 0.001, 0.002,        0.02]}, # fraction per year
                       'elite_migration2': {'migration_fraction_elites':          [0.002, 0.008, 0.02, 0.04]}, # fraction per year (stock, modest, large, extra large increase)
                       'elite_misery':    {'base_misery_threshold': [.65, .75, .80, .85, .90, .95]}, # fraction of elite opportunity consumed
                       'cavalry_diffusion_time': {'mounted_warfare_diffusion_time': [500,600,700,800]}, # years
                       'NC_trigger_pop':  {'confed_trigger_population': [4,5,6,7,8]}, # millions (depends on camels!) [6.6,6.8,7.0,7.2,7.4] [8,9,10,11,12,13]
                       # Typical value assumes max size of confed is 1/2 steppe of 470 regions; tribes are about 30 regions.
                       # Also assumes max AS pop is around 200M and trigger is 6.5M so we expect to add 1.6reg per M or 1 tribe per 19M
                       # if the scale factor is smaller (1.2) then it is 1 tribe per 25M so slower than usual
                       'NC_tribal_scale': {'confed_tribal_scale': [1.6,1.7,1.8,1.9,2.0], # [1.3,1.4,1.5,1.6], # lower causes slower NC growth
                                           'CONSTANTS': ['confed_trigger_population = 6.0']}, # lower trigger permits later NCs to rise easier
                       'max_s':           {'max_polity_s': [1.9, 2.1, 4.0]},# [1.6, 1.7, 1.8, 1.9, 2.1, 4.0]
                       'nt_nc_s':         {'max_tribal_s': [1.20, 1.43, 1.80, 1.43, 1.43],
                                           'max_polity_s': [2.10, 2.10, 2.10, 2.60, 4.00]},
                       'strike_zones': {'tribal_strike_distance': [100, 200, 300, 400, 500, 600], # km
                                        'confed_strike_distance': [200, 400, 600, 800, 800, 900], # km
                                        },
                       's_adapt': {'s_adapt_factor': [0.01, 0.03, 0.05, 0.07],},
                       # Get data for Figure 9/10
                       'sea_desert_cross':   {'sea_eff':    [0.0, 0.3, 0.3, 0.0],
                                              'desert_eff': [0.0, 0.3, 0.0, 0.3],
                                              'CONSTANTS': sea_desert_consts,
                                              },
                       'sea_desert':   {'sea_eff':    [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                        'desert_eff': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                        'CONSTANTS': sea_desert_consts,
                                        },
                       'desert_sweep':   {'sea_eff':    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # no change except distance
                                          'desert_eff': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                          'CONSTANTS': sea_desert_consts,
                                        },
                       }) 
