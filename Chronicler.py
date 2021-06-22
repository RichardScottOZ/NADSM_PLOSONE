import os
import os.path
import shutil
import time
import numpy as np
import math
from Constants import *
time_loading_matplotlib = False
load_time = time.time();
import matplotlib # this loads quickly
# change show() to savefig()
matplotlib.use('Agg') # this backend produces PNG figures but w/o the interaction (faster)
matplotlib.interactive(False);
# matplotlib.use('agg') # avoid display and requirement of X server
matplotlib.rcParams['mathtext.fontset'] = 'cm' # this speeds (halves) the load of matplotlib; probably a font problem
matplotlib.rcParams['font.size'] = 8 # smaller font
# Add more space between the plat and the xticj and ytick labels
# matplotlib.rcParams['xtick.major.pad'] = 10
# matplotlib.rcParams['ytick.major.pad'] = 10
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt # TAKES LOTS OF TIME on first load
if time_loading_matplotlib:
    print('Loading matplotlib took %.1fs' % (time.time() - load_time))
from Utils import *
from Constants import * # GEO_x and POLITY_x

display_area_size = True # CONTROL True displays in km^2, False in regions

trial_markers = '.+odxs'

def lose_spines(ax):
    # Turn off spines for a plot
    ax.spines['left'  ].set_visible(False) # True
    ax.spines['right' ].set_visible(False)
    ax.spines['top'   ].set_visible(False)
    ax.spines['bottom'].set_visible(False) # True

def color_background(ax):
    # Set background to a light grey to help the blue and green points stand out
    # Helps draw the eye between subplots on same figure
    ax.patch.set_facecolor((220./255.,220./255.,220./255.))
    # UNNEEDED ax.patch.set_alpha(0.5)
    
# A base class mixed in with WHM and its subclasses to handle displays and writing of state
class Chronicler(object):
    def initialize_parameters(self):
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


    def initialize_chronicler(self):
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

    def setup_chronicler_for_trial(self):
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
        
    def chronicle(self):
        if (self.this_year % self.display_time_interval) == 0:
            self.update_chronicle()
            if self.display_show_figures:
                self.update_display()
            
    def update_chronicle(self):
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

    def finalize_chronicler(self):
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
    def initialize_display(self):
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

    def update_display_for_trial(self):
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
            

    def update_display_figure_hook(self, axis):
        pass
    
    def update_display(self):
        print('%8s' % self.this_pretty_year)
        #DEAD self.predicted_years.append(self.this_year) # TODO update in chronicle
        # update figures
        geo = self.geography

        ax_p = self.display_ax_p
        ax_p.cla()
        lat = geo.lat
        lon = geo.lon
        # appears that the default . marker has fillstyle:None
        ax_p.plot(lon,lat,marker='.',linestyle='None',color='Silver',label='_nolegend_',zorder=1)
        ax_p.set_xlabel("Longitude")
        ax_p.set_ylabel("Latitude")
        # Use this rather than max/min so show where we selected (see India)
        lon_w,lon_e,lat_s,lat_n = geo.bounding_box
        ax_p.set_xlim(lon_w,lon_e)
        ax_p.set_ylim(lat_s,lat_n)
        ax_p.grid(True) # both axes vs. ax.xaxis.grid(True) or ax.yaxis.grid(True)
        for polity in self.polities:
            regions = polity.display_regions # from update_chronicle(), which tests large_enough() etc.
            if regions:
                ax_p.plot(lon[regions],lat[regions],
                          linestyle='None',
                          marker=polity.flag['marker'], markeredgewidth=0,
                          color=polity.flag['color'],
                          label=polity.name if polity.name else '_nolegend_',zorder=2)
        ax_p.set_title(r"%s %s" % (self.title_tag,self.this_pretty_year))
        self.update_display_figure_hook(self.display_ax_p)
        self.display_fig_p.canvas.draw() # force out drawing

        if self.display_make_movie:
            self.display_counter += 1
            frame_name = os.path.join(self.chronicler_directory,
                                      '%s_%05d.png' % (self.fname_prefix,self.display_counter))
            self.display_fig_p.savefig(frame_name, format='png')

        # from update_chronicle() 
        all_polities,n_polities,ignore,km2_under_polity,ignore,ag_km2_under_polity = self.current_predicted_values

        if self.track_polity_biomes is not agri_biomes: # there are others?
            self.display_ax_a.plot(self.this_year,km2_under_polity, marker=self.trial_marker, color='Red')
        self.display_ax_a.plot(self.this_year,ag_km2_under_polity, marker=self.trial_marker, color='Magenta')
        self.update_display_figure_hook(self.display_ax_a)

        self.display_fig_a.canvas.draw() # force out drawing
        if True: # Pretty size labels?
            yticks = self.display_ax_a.get_yticks().tolist()
            if yticks != self.display_fig_a_yticks: # only refresh labels when they change
                self.display_fig_a_yticks = yticks # update cache
                # update the ylabels
                labels = [millions(regions) for regions in yticks]
                self.display_ax_a.set_yticklabels(labels)
                self.display_fig_a.canvas.draw() # force out again

        if False:
            self.display_ax_e.plot(self.this_year,n_polities, marker=self.trial_marker,color='Blue')
        self.display_ax_e.plot(self.this_year,all_polities, marker=self.trial_marker,color='Red')
        self.update_display_figure_hook(self.display_ax_e)
        self.display_fig_e.canvas.draw() # force out drawing
        if False: # typically the figures are saved and the Agg backend complains if show() is used instead of savefig()
            plt.show(block=False)
            plt.pause(0.2) # CRITICAL to forcing the window up!
        if False: # DEBUG
            print("%d HX: %d %d %d" % (self.this_year,n_polities,all_polities,self.predicted_regions_under_polity[-1]))
        return True

    def finalize_display(self):
        if self.display_make_movie:
            movie_name = os.path.join(self.chronicler_directory,trial_basename(self,'History%s.mp4'))
            # Use this UNIX utility to create a movie from the frame images
            # normally in /opt/local/bin/ but could be in /usr/local/bin
            cmd = 'ffmpeg -loglevel quiet -r %s -i %s/%s_%%05d.png -c:v libx264 -pix_fmt yuv420p %s'
            run_cmd_shell(cmd % (self.display_movie_fps, self.chronicler_directory, self.fname_prefix, movie_name))
            # clean up the intermediate files
            run_cmd_shell('rm -rf %s/%s_*.png' % (self.chronicler_directory,self.fname_prefix))

        if False and self.geography.actual_history:
            # compare the predicted to the historical data
            # Move this to finalize_chronicler() and dump it
            # DEAD remove this
            print("HX_Years = %s;" % self.geography.years)
            print("HX_Yearly_polity_count = %s;" % self.historical_polity_count)
            print("HX_Cumulative_polity_count = %s;" % self.historical_cumulative_polity_count)
            print("HX_Yearly_regions_under_polity = %s;" % self.historical_regions_under_polity)
            print("PX_Years = %s;" % self.predicted_years)
            print("PX_Yearly_polity_count = %s;" % self.predicted_polity_count)
            print("PX_Cumulative_polity_count = %s;" % self.predicted_cumulative_polity_count)
            print("PX_Yearly_regions_under_polity = %s;" % self.predicted_regions_under_polity)
            
        if self.display_save_figures:
            self.display_fig_p.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'World%s.png')),format='png') # final world
            if self.display_composite_figure == 0:
                self.display_fig_a.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'Area%s.png')),format='png')
                self.display_fig_e.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'Polity%s.png')),format='png')
            elif self.display_composite_figure == 1:
                self.display_fig_a.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'Data%s.png')),format='png')
                self.display_fig_e.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'Polity%s.png')),format='png')
            elif self.display_composite_figure == 2:
                self.display_fig_a.savefig(os.path.join(self.chronicler_directory,trial_basename(self,'Data%s.png')),format='png')
            else:
                raise RuntimeError("Unknown type of figure (%d)" % self.display_composite_figure)
            

        if self.trial == self.n_trials-1:
            # Close after the last trial
            plt.close(self.display_fig_p)
            plt.close(self.display_fig_a)
            plt.close(self.display_fig_e)
            return True
        else:
            return False
    
    
    def set_pretty_years(self,fig,ax):
        # Assumes the xaxis is in 'years'
        # This converts it to a pretty form
        # you must call draw() first to set the locations
        ax.set_xlim(self.first_year,self.last_year)
        fig.canvas.draw() # force out drawing
        if self.display_composite_figure:
            pass # keep as numbers for size
        else:
            labels = [pretty_year(loc,padded=False) for loc in ax.get_xticks()]
            ax.set_xticklabels(labels)
            ax.set_xlabel("Year")
            fig.canvas.draw() # force out again
