# compute and display R2 statistics about how the underlying WHM is doing against historical actuals
# mixin with the WHM of your choice
from Constants import * # GEO_x and POLITY_x
from WHM import *
whm = None # package global
import numpy as np
import copy
from Utils import *

# Control whether we compute and track R2 values

# There are two things, intermixed.  The first is whether you want the scaled
# prediction (and observation, if actual_history) data dumped. Then there is
# whether you want the comparison shown and saved.  The later can only happen if
# there there is actual_history.  The first bit is controlled by dump_R2_data;
# the second by enable_R2_comparison.  What gets dumped depends on what is
# available.
# NOTE: we always collect the prediction data.  Whether we dump and compare it is separate.

# Dump the data for density graphs
dump_predicted_history = True
dump_R2_data = True
dump_R2_observations = True # requires dump_R2_data == True; dump_R2_observations could be False to save space in trial.mat file

# enable_R2_comparison can be disabled here but we assume it is always on
# It is disabled if there is no actual history to compare against
# initialize at setup_for_trial()
# clear that array during R2 calcs each time step after incorporation
# update during state_of_the_union if alive and size is big enough and NOT hinterland or unoccupied
enable_R2_comparison = True
# Peter observes that the value should be reported as if the year is middle of the interval
# rather than the end...
report_R2_middle = True

class ComparablePolity(object):
    def state_of_the_union(self):
        super().state_of_the_union()
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
class CompareWithHistory(object):

    def initialize_parameters(self):
        global whm
        whm = self # set package global
        super().initialize_parameters()

    def initialize_chronicler(self):
        if not self.geography.actual_history:
            global enable_R2_comparison
            enable_R2_comparison = False
        super().initialize_chronicler()
    
    def initialize_display(self):
        super().initialize_display()
        global enable_R2_comparison
        if not enable_R2_comparison:
            return # no display needed
        
        # Spatial R2 metric per display time
        if self.display_composite_figure in (0,1):
            fig_R2 = plt.figure()
            ax_R2 = fig_R2.add_subplot(111)
            ax_R2.set_title("%s $R^2 (polity size >= %d)$" % (self.title_tag,self.compare_polity_size))
        elif self.display_composite_figure == 2:
            fig_R2 = self.display_fig_a # use master figure
            ax_R2 = fig_R2.add_subplot(224)
            ax_R2.yaxis.tick_right() # move ylabel to right
            #DEAD ax_R2.yaxis.set_label_position("right")
            # avoid title

        self.display_fig_R2 = fig_R2
        self.display_ax_R2 = ax_R2
        lose_spines(ax_R2)
        if self.display_composite_figure > 0:
            color_background(ax_R2)

        ax_R2.set_ylabel("Spatial $R^2$")
        ax_R2.grid(True) # both axes vs. ax.xaxis.grid(True) or ax.yaxis.grid(True)
        ax_R2.set_ylim(0.0,1.0)
        self.set_pretty_years(fig_R2,ax_R2)
        
    def setup_for_trial(self):
        super().setup_for_trial()
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

    def update_display_for_trial(self):
        super().update_display_for_trial()
        if self.display_composite_figure:
            pass
        else:
            self.display_ax_R2.set_title("%s $R^2 (polity size >= %d)$" % (self.title_tag,self.compare_polity_size))


    def finalize_display(self):
        global enable_R2_comparison
        if not enable_R2_comparison:
            return super().finalize_display()
        
        if self.display_save_figures:
            if self.display_composite_figure == 0:
                self.display_fig_R2.savefig(os.path.join(self.chronicler_directory,'R2.png'),format='png')


        # super (return True if figures should be closed?)
        if super().finalize_display():
            plt.close(self.display_fig_R2)
            return True
        else:
            return False
        
    def state_of_the_world(self):
        super().state_of_the_world()
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
        
