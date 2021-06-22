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
    def initialize_chronicler(self):
        # do this once for this entire run
        if self.geography.actual_history:
            self.historical_population = self.geography.total_population
        else:
            self.historical_population = None
        super().initialize_chronicler()

    def initialize_display(self):
        # if we get here we need this (and Chronicler will have loaded it)
        super().initialize_display()
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
        super().setup_chronicler_for_trial()
        
    def update_display_for_trial(self):
        super().update_display_for_trial()
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
        super().update_chronicle()

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
            
        super().finalize_chronicler()
            
        
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
