import os
import os.path
import time
import numpy as np
import math
from Constants import * # GEO_x and POLITY_x
from Chronicler import *
# Chronicler just above will import this first so it doesn't take any time here
import matplotlib.pyplot as plt
report_regional_population = True # if True dumps population per region at geo.year intervals (1.5Mb files for OldWorld)
# See Population
class PopChronicler(Chronicler):
##f Chronicler:chronicle(...)
##f Chronicler:initialize_parameters(...)
##f Chronicler:set_pretty_years(...)
##f Chronicler:update_display_figure_hook(...)
    def initialize_chronicler(self):
        # do this once for this entire run
        if self.geography.actual_history:
            self.historical_population = self.geography.total_population
        else:
            self.historical_population = None
##s Chronicler.py:Chronicler:initialize_chronicler
        if self.display_make_movie:
            self.display_show_figures = True # force this on for movies

        # Potential BUG if you run multiple trials you will get a single movie that shows all trials
        if self.data_subdir is None: # permit code override
            self.data_subdir = "%s_%s" % (self.name, self.options.t if self.options.t else self.geography_name)
            for experiment in self.options.X:
                self.data_subdir += '_%s' % experiment
        self.data_subdir = ensure_basename(self.data_subdir)
        chronicler_directory = os.path.join(os.path.abspath(self.options.o),self.data_subdir)
        shutil.rmtree(chronicler_directory,ignore_errors=True)
        try:
            os.makedirs(chronicler_directory)
        except:
            raise RuntimeError("Unable to make chronicler directory %s" %  chronicler_directory)
        else:
            print("Saving figures and chronicler data to %s" %  chronicler_directory)
            self.chronicler_directory = chronicler_directory

        # Do this once
        historical_polity_count = []
        historical_cumulative_polity_count = []
        historical_regions_under_polity = []
        historical_km2_under_polity = []        
        historical_ag_regions_under_polity = []
        historical_ag_km2_under_polity = []
        geo = self.geography
        if geo.actual_history:
            print("Computing historical statistics...")
            history = geo.history
            Biome  = geo.Biome
            polities_to_date = set() # set() not critical to running order
            ag_i = np.where(Biome == GEO_AGRICULTURAL)[0]
            self.total_agricultural_regions = len(ag_i)
            self.total_agricultural_km2 = np.sum(self.area_km2[ag_i])
            track_biomes_i = np.array([r for r in range(self.geography.n_regions) if Biome[r] in self.track_polity_biomes])
            for year_i in range(len(self.geography.years)):
                polities_this_year = []
                polity_regions_this_year = []
                for polity_id in unique(history[:,year_i]):
                    # skip 'hinterland' (polity id = 0)
                    # open-coded version of large_enough() that applies to history data, not generated polites
                    # BUG: if we are running agricultural only we need to filter polity if it has any regions with Biome == GEO_AGRICULTURAL
                    # the problem is that when we want to initialize_display() we have not yet set parameters whether we care about ag only or not
                    if polity_id:
                        polity_i = np.where(history[:,year_i] == polity_id)[0]
                        if len(polity_i) >= self.compare_polity_size: # large_enough() in history?
                            polities_this_year.append(polity_id)
                            polity_regions_this_year.extend(polity_i.tolist())

                historical_polity_count.append(len(polities_this_year)) 
                polities_to_date = polities_to_date.union(polities_this_year) # unique polities to date
                historical_cumulative_polity_count.append(len(polities_to_date)) # account for hinterland

                polity_regions_this_year = np.array(polity_regions_this_year)
                regions_this_year = np.intersect1d(polity_regions_this_year,track_biomes_i)
                historical_regions_under_polity.append(len(regions_this_year))
                historical_km2_under_polity.append(np.sum(self.area_km2[regions_this_year]))

                regions_this_year = np.intersect1d(polity_regions_this_year,ag_i)
                historical_ag_regions_under_polity.append(len(regions_this_year))
                historical_ag_km2_under_polity.append(np.sum(self.area_km2[regions_this_year]))

        self.historical_years = self.geography.years
        self.historical_polity_count = historical_polity_count
        self.historical_cumulative_polity_count = historical_cumulative_polity_count
        self.historical_regions_under_polity = historical_regions_under_polity
        self.historical_km2_under_polity = historical_km2_under_polity
        self.historical_ag_regions_under_polity = historical_ag_regions_under_polity
        self.historical_ag_km2_under_polity = historical_ag_km2_under_polity

        # BUG: we count regions and scale by a constant for area
        # really we should record the actual indicies and multiply by area_km2, which is available, then sum
        # if we want regions multiply by ones(len(self.territories),1) rather than area_km2
        if display_area_size:
            # This is an approximation since every region needs scaling by cos(geo.lat)
            self.region_units_scale = geo.equatorial_region_km2
        else:
            self.region_units_scale = 1;

        if self.display_show_figures:
            self.initialize_display()

##e Chronicler.py:Chronicler:initialize_chronicler

    def initialize_display(self):
        # if we get here we need this (and Chronicler will have loaded it)
##s Chronicler.py:Chronicler:initialize_display
        # BAD BUG this is not called after setup_for_trial() for a trial but during infrastructure setup
        # As a consequence if any parameters like last_year, etc. are changed during setup_for_trial()
        # they aren't reflected in the figures (like xlim)
        # Same with title_tag, etc.
        
        # NOTE: To add greek characters in formatted statements use \\Delta= etc. and we add the $'s here to interpret them
        # If you want spaces to show in parameter_description, you should quote them using \ since it is rendered mathtext
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['lines.markersize'] = 4 # 6 is default
        self.title_tag = "%s\n%s" % (self.name,self.geography_name) # initial version
        geo = self.geography
        lat = geo.lat
        lon = geo.lon

        # Compute page dimensions so OldWorld would fit on a 6x6
        # We do this rather than set the page size
        # to get a nice looking compact image
        # TODO permit the following parameters that override
        # p.display.lat_page_size (None)
        # p.display.lon_page_size (None)
        lat_page_size = math.ceil((max(lat) - min(lat))/20.0)
        lon_page_size = math.ceil((max(lon) - min(lon))/20.0)

        # Polity location each display time
        # set DPI so we get an even number of pixels (for ffmpeg)
        # fig_p = plt.figure(dpi=100,figsize=(page_size,page_size))
        fig_p = plt.figure(dpi=100,figsize=(lon_page_size,lat_page_size))
        self.display_fig_p = fig_p 
        ax_p = fig_p.add_subplot(111)
        self.display_ax_p = ax_p
        ax_p.set_aspect('equal')
        lose_spines(ax_p)

        if display_area_size:
            region_scale = "$km^2$"
        else:
            region_scale = "regions"
                    
        # Polity area per display time
        if self.display_composite_figure == 0:
            fig_a = plt.figure()
            ax_a = fig_a.add_subplot(111)
        elif self.display_composite_figure == 1:
            # Master figure with 2 panels in landscape mode
            # The 4 panel figure has a good aspect ratio
            # But we 2 panel version has to be at least 5 inches to avoid cutting off the title
            # the long version needs to scale accordingly
            fig_a = plt.figure(dpi=100,figsize=(11*6.5/5,5))
            ax_a = fig_a.add_subplot(121) # Area on left
        elif self.display_composite_figure == 2:
            # Master figure with 4 panels in landscape mode
            fig_a = plt.figure(dpi=100,figsize=(11,6.5)) 
            ax_a = fig_a.add_subplot(221) # Area in upper left
        else:
            raise RuntimeError("Unknown type of figure (%d)" % self.display_composite_figure)

        self.display_fig_a = fig_a
        self.display_ax_a = ax_a
        lose_spines(ax_a)
        if self.display_composite_figure > 0:
            color_background(ax_a)
        ax_a.set_ylabel("Area (%s) under polities size >= %d" % (region_scale,self.compare_polity_size)) # filter state
        ax_a.grid(True) # both axes vs. ax.xaxis.grid(True) or ax.yaxis.grid(True)
        ax_a.set_title("%s Area" % self.title_tag)
        if len(self.historical_km2_under_polity):
            ax_a.plot(self.geography.years,np.array(self.historical_km2_under_polity),color='Black',
                      label='%s(Turchin et al, 2013)' % self.tracking_biomes)
        if True and len(self.track_polity_biomes) > 1:
            ax_a.plot(self.geography.years,np.array(self.historical_ag_km2_under_polity),
                      color='Blue',linestyle='-', label='Ag regions(Turchin et al, 2013)')
            if False:
                # The idea where was to show how much of the theater *could* have been filled with states under ag vs. what happened in the time frame simulated
                # Nice idea but flawed since it doesn't reflect the actual sizes of the regions
                # but then the plot expression above doesn't either...
                ax_a.plot([self.geography.years[0], self.geography.years[-1]],
                          np.array([self.total_agricultural_km2,self.total_agricultural_km2]),
                          color='Green',linestyle='--',label='Total available agricultural area')
        self.set_pretty_years(fig_a,ax_a)
        self.display_fig_a_yticks = [] # initialize yticks cache


        # Number of polities per display time
        if self.display_composite_figure in (0,1):
            fig_e = plt.figure()
            ax_e = fig_e.add_subplot(111)
            ax_e.set_title("%s Polities" % self.title_tag)
        elif self.display_composite_figure == 2:
            fig_e = self.display_fig_a # use master figure
            ax_e = fig_e.add_subplot(223)
            # avoid title

        self.display_fig_e = fig_e
        self.display_ax_e = ax_e
        lose_spines(ax_e)
        if self.display_composite_figure > 0:
            color_background(ax_e)

        ax_e.set_ylabel("Number of polities size >= %d" % self.compare_polity_size)
        ax_e.grid(True) # both axes vs. ax.xaxis.grid(True) or ax.yaxis.grid(True)
        if len(self.historical_polity_count):
            if False:
                ax_e.plot(self.geography.years,self.historical_polity_count,
                          color='Green',linestyle='-', label='per year (Turchin et al, 2013)')
            ax_e.plot(self.geography.years,self.historical_cumulative_polity_count,
                      color='Black',linestyle='-',label='cumulative (Turchin et al, 2013)')
        self.set_pretty_years(fig_e,ax_e)

##e Chronicler.py:Chronicler:initialize_display
        # Polity population per display time
        if self.display_composite_figure == 0:
            fig_pop = plt.figure()
            ax_pop = fig_pop.add_subplot(111)
            ax_pop.set_title("%s %s Population" % (self.name,self.geography_name))
        elif self.display_composite_figure == 1:
            # Master figure with 2 panels in landscape mode
            fig_pop = self.display_fig_a # use master figure
            ax_pop = fig_pop.add_subplot(122) # Population on right
            ax_pop.yaxis.tick_right() # move ylabel to right
            #DEAD ax_pop.yaxis.set_label_position("right")
            # Shorten the title when composite
            ax_pop.set_title("%s Population" % self.geography_name)
        elif self.display_composite_figure == 2:
            # Master figure with 4 panels in landscape mode
            fig_pop = self.display_fig_a # use master figure
            ax_pop = fig_pop.add_subplot(222) # upper right
            ax_pop.yaxis.tick_right() # move ylabel to right
            #DEAD ax_pop.yaxis.set_label_position("right")
            # Shorten the title when composite
            ax_pop.set_title("%s Population" % self.geography_name)

        self.display_fig_pop = fig_pop
        self.display_ax_pop = ax_pop
        lose_spines(ax_pop)
        if self.display_composite_figure > 0:
            color_background(ax_pop)
        # TODO move this display to update_display_for_trial()
        if self.historical_population is not None:
            # TODO add to mat file:  self.geography.total_population as HX_population 
            # self.historical_population = self.geography.total_population 
            self.display_ax_pop.plot(self.geography.years,self.historical_population,color='Black',label='Kaplan, 2014')
        ax_pop.set_ylabel("Population")
        ax_pop.grid(True) # both axes vs. ax.xaxis.grid(True) or ax.yaxis.grid(True)
        self.set_pretty_years(fig_pop,ax_pop)
        self.display_fig_pop_yticks = [] # initialize yticks cache

    def setup_chronicler_for_trial(self):
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
        
    def update_display_for_trial(self):
##s Chronicler.py:Chronicler:update_display_for_trial
        # initialize for possible comparison later
        # update titles with trial information (via updated trial_tag)
        if self.display_make_movie:
            # in the case of multiprocessing, tag all frames with trial number
            # do this here because trial is finally set
            self.fname_prefix = trial_basename(self,'f%s') 

        self.display_ax_p.set_title(r"%s %s" % (self.title_tag,self.this_pretty_year))
        self.display_ax_a.set_title("%s Area" % self.title_tag)
        if self.display_composite_figure:
            pass
        else:
            self.display_ax_e.set_title("%s Polities" % self.title_tag)
            

##e Chronicler.py:Chronicler:update_display_for_trial
        if not self.display_composite_figure:
            # Update title only if solo figure
            self.display_ax_pop.set_title("%s %s Population" % (self.name,self.geography_name))

    def compute_world_population(self,report_status=False):
        world_population = sum(self.population)
        alive_polities = len([p for p in self.polities if p.state is POLITY_ALIVE])
        dead_polities = len(self.polities) - alive_polities;
        if report_status:
            print(('Alive: %d/%d World pop: %.3fM Hinterland pop: %.3fM (%d regions)' % # DEBUG
                   (alive_polities,
                    dead_polities,
                    world_population/million,
                    self.Hinterland.total_population/million,
                    len(self.Hinterland.territories))))
        return (world_population,alive_polities,dead_polities)

    def compute_world_population_OLD(self,report_status=False):
        # map over polities and sum population over territories
        # reporting status slows down execution because of output
        world_population = 0
        alive_polities = 0
        dead_polities = 0
        for polity in self.polities:
            # assert polity.state is not POLITY_ALIVE and polity.total_population == 0
            # Now count number of alive and dead polities according to size limit
            if polity.state is POLITY_ALIVE:
                # This should include self.Hinterland.total_population too
                alive_polities += 1
                world_population += polity.total_population
                if report_status:
                    n_t_ag = len([t for t in polity.territories if t.Biome == GEO_AGRICULTURAL])
                    print(('%s: %s %d(%d) %.3fM %.2f%%' % # DEBUG
                           (self.this_pretty_year,
                            polity.name,
                            len(polity.territories),n_t_ag,
                            polity.total_population/million,
                            (polity.total_population/polity.total_k)*100.0)))
            else:
                dead_polities += 1

        if report_status:
            print(('Alive: %d/%d World pop: %.3fM Hinterland pop: %.3fM (%d regions)' % # DEBUG
                   (alive_polities,
                    dead_polities,
                    world_population/million,
                    self.Hinterland.total_population/million,
                    len(self.Hinterland.territories))))
        return (world_population,alive_polities,dead_polities)
        
    def update_chronicle(self):
        # TODO have it compute and report predicted k as well, including hinterland
        # Why don't we just sum(self.population) here?  and sum(self.k)?  Clearly these are alive values
        (world_population,alive_polities,dead_polities) = self.compute_world_population(report_status=False)
        # DEAD world_population = sum(self.population);
        if self.report_regional_population:
            residual_population = world_population
            report = 'Population: %d %.1f %.1f' % (self.this_year,np.sum(self.k),world_population)
            for region in self.region_list:
                region_population = sum(self.population[self.region_territories_i[region]])
                report = '%s %.1f' % (report,region_population)
                residual_population -= region_population
            print('%s %.1f' % (report, residual_population))
        self.world_population = world_population # for update_display below
        if self.this_year in self.geography.years:
            # Record these statistics on the original sampling grid
            self.predicted_population.append(world_population)
            if self.region_population is not None:
                self.region_population[:,self.geography.years.index(self.this_year)] = self.population
##s Chronicler.py:Chronicler:update_chronicle
        dump_cum_polities = False
        regions_under_polity = 0
        km2_under_polity = 0
        ag_regions_under_polity = 0
        ag_km2_under_polity = 0
        n_polities   = 0 # currently alive
        all_polities = 0 # ever alive and big enough (ever) DEAD?
        if dump_cum_polities:
            print('Cum %s:' % self.this_pretty_year)
        # BIG BUG:  At each display_time_interval we runs this code, which looks at *all* the polities
        # created to date that are large enough and long enough to count and keeps track of their increasing count
        # This keeps track of the right number of 'alive' polities at each step.
        # However, for the cumulative number we can about which those unique polities that were alive at a century boundary
        # for this we need to (1) ensure display_time_interval is 100 years and (2) keep track of previous alive
        # polities at that point and add to the list only *new* alive ones
        #
        alive_polities = []
        for polity in self.polities:
            polity.display_regions = None
            # NOTE: if whm.compare_polity_size is large (e.g., 10) and we are dissolving in Actual*
            # then the count we get, in spite of flipping the regions at the proper rate,
            # might not yield the right number of regions to display and be counted
            # As a consequence, things like the area (green dots) will be below the predicted
            # Reducing the number to 1 or even 0 helps correct that, although it can still be off
            # since it is stochastic
            if not polity.large_enough(dead_or_alive=True):
                continue # don't count
            # This polity was large enough to count sometime (dead or alive)
            # Check was it ever old enough to count as part of all_polities statistic (vs. historical century old data)
            if polity.end_year - polity.start_year >= self.compare_polity_duration:
                all_polities += 1
                if dump_cum_polities:
                    print(' %d %s' % (polity.id,polity_state_tags[polity.state]))
            if polity.state is not POLITY_ALIVE:
                continue # don't count
            # currently alive and counting
            n_polities += 1
            alive_polities.append(polity)
            regions_under_polity += polity.size
            km2_under_polity += polity.km2_size
            regions = [] # somewhat expensive but infrequent
            ag_territory_i = [] # same as agricultural_i if asserted but just in case
            for territory in polity.territories:
                t_regions = territory.regions
                if territory.Biome in self.track_polity_biomes:
                    regions_under_polity += len(t_regions)
                if territory.Biome == GEO_AGRICULTURAL:
                    ag_territory_i.append(territory.t_index)
                    ag_regions_under_polity += len(t_regions)
                regions.extend(t_regions)
            polity.display_regions = regions
            ag_km2_under_polity += np.sum(self.area_km2[ag_territory_i])

        self.current_predicted_values = (all_polities,n_polities,
                                         regions_under_polity,km2_under_polity,
                                         ag_regions_under_polity,ag_km2_under_polity)
        if self.this_year in self.geography.years:
            # Record these statistics on the original sampling grid
            self.predicted_years.append(self.this_year)
            self.predicted_cumulative_polities.extend(alive_polities)
            self.predicted_cumulative_polities = unique(self.predicted_cumulative_polities)
            self.predicted_cumulative_polity_count.append(len(self.predicted_cumulative_polities))
            self.predicted_polity_count.append(n_polities)            
            self.predicted_regions_under_polity.append(regions_under_polity)
            self.predicted_km2_under_polity.append(km2_under_polity)
            self.predicted_ag_regions_under_polity.append(ag_regions_under_polity)
            self.predicted_ag_km2_under_polity.append(ag_km2_under_polity)

##e Chronicler.py:Chronicler:update_chronicle

    def update_display(self):
        if super().update_display():
            self.display_ax_pop.plot(self.this_year,self.world_population,
                                     marker=self.trial_marker, color='Red',label='Predicted')
            self.update_display_figure_hook(self.display_ax_pop)
            self.display_fig_pop.canvas.draw() # force out drawing
            yticks = self.display_ax_pop.get_yticks().tolist()
            if yticks != self.display_fig_pop_yticks: # only refresh labels when they change
                self.display_fig_pop_yticks = yticks # update cache
                # update the ylabels
                labels = [millions(population) for population in yticks]
                self.display_ax_pop.set_yticklabels(labels)
                self.display_fig_pop.canvas.draw() # force out again
            return True
        else:
            return False


    def finalize_chronicler(self):
        # Order is pleasant for statistics generation
        (world_population,alive_polities,dead_polities) = self.compute_world_population(report_status=False)
        print(('Finished in %s: %d polities (A:%d/D:%d >= %d) World pop: %.3fM Hinterland pop: %.3fM (%d regions)' % # DEBUG
               (self.this_pretty_year,
                len(self.polities), alive_polities, dead_polities,self.compare_polity_size,
                world_population/million,
                self.Hinterland.total_population/million,
                len(self.Hinterland.territories))))
        if self.historical_population is not None:
            self.trial_data['HX_population'] = self.historical_population
        self.trial_data['PX_Yearly_population'] = self.predicted_population
        if self.region_population is not None:
            self.trial_data['region_population'] = self.region_population
            
##s Chronicler.py:Chronicler:finalize_chronicler
        #DEAD super().finalize_chronicler()

        # Really, reimplement the stats graphs
        if False: # DEBUG show states alive at the end of time
            print('Polities alive at the end of history:')
            for polity in self.polities:
                if polity.quasi_state:
                    continue

                if polity.state is POLITY_ALIVE:
                    print(' %s(%dy %d/%d)' % (polity,self.this_year - polity.start_year,polity.size,polity.max_size)) 
        if False: # DEBUG show all states created
            print('Polities created to the end of history:')
            for polity in self.polities:
                if polity.quasi_state:
                    continue
                if polity.state is POLITY_ALIVE:
                    end_year = self.this_year
                else:
                    end_year = polity.end_year
                print("%d %d %d %d %d" % (polity.id,polity.start_year,end_year,polity.size,polity.max_size))

        if self.display_show_figures:
            self.finalize_display()

        if True:
            import scipy.io as sio
            trial_data = self.trial_data
            trial_data['HX_Years'] = self.geography.years
            trial_data['HX_Yearly_polity_count'] = self.historical_polity_count
            trial_data['HX_Cumulative_polity_count'] = self.historical_cumulative_polity_count
            trial_data['HX_Yearly_regions_under_polity'] = self.historical_regions_under_polity
            trial_data['HX_Yearly_km2_under_polity'] = self.historical_km2_under_polity
            trial_data['HX_Yearly_ag_regions_under_polity'] = self.historical_ag_regions_under_polity
            trial_data['HX_Yearly_ag_km2_under_polity'] = self.historical_ag_km2_under_polity            

            trial_data['PX_Years'] = self.predicted_years
            trial_data['PX_Yearly_polity_count'] = self.predicted_polity_count
            trial_data['PX_Cumulative_polity_count'] = self.predicted_cumulative_polity_count
            trial_data['PX_Yearly_regions_under_polity'] = self.predicted_regions_under_polity
            trial_data['PX_Yearly_km2_under_polity'] = self.predicted_km2_under_polity
            trial_data['PX_Yearly_ag_regions_under_polity'] = self.predicted_ag_regions_under_polity
            trial_data['PX_Yearly_ag_km2_under_polity'] = self.predicted_ag_km2_under_polity
            # Add parameters to trial data
            for tag in print_scalar_members(self,print_values=False):
                trial_data[tag] = getattr(self,tag,'')
            trial_data['unfold_time'] = time.time() - self.start_time # seconds
            sio.savemat(os.path.join(self.chronicler_directory,trial_basename(self,'trial%s.mat')), trial_data)


    # [Display]
##e Chronicler.py:Chronicler:finalize_chronicler
            
        
    def finalize_display(self):
        if self.display_save_figures:
            if self.display_composite_figure == 0:
                self.display_fig_pop.savefig(os.path.join(self.chronicler_directory,'Population.png'),format='png')
            else: # included in other composite figures
                pass

        # super (return True if figures should be closed?)
        if super().finalize_display():
            plt.close(self.display_fig_pop)
            return True
        else:
            return False
