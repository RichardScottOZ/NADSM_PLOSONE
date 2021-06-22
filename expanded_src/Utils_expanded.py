# Support routines for WHM
import sys
import traceback
import os
import os.path
from stat import * # for chmod
import time
import cProfile,pstats
import random
import subprocess
import shutil
import collections
import argparse
import numpy as np
from Constants import * # use_set_versions
import pprint

world = None

# To preserve our ability to deterministically debug an essentially stochastic program
# we need to be reset the random number generator seed and then, because of bad explicit
# or implicit choice of datastructures that rely on memory addresses, we might not replicate
# even with the same seed.  So here for debugging we provide a set of wrapper functions,
# normally disabled, that wrap all the random package functions and count calls.  The count
# can be accessed (esp in the debugger) while we bracket the locations (via print statements)
# where the repeated runs run off the rails between one another.
# You might need to import Utils into pdb to access Utils.random_number_debug_counter
# In ADSM.py try turning on dump_stats to get closer to the calls that mess up

# Known issues:
# - anything in python that uses memory addresses as keys will not preserve canonical order even if unique
# - thus the use of sets rather than lists, requesting keys() from dicts, etc and expecting the same order is flawed
#   but if the values are indicies that is ok since order doesn't matter there
# - numpy via union1d, intersect1d, setdiff1d are fast BUT use memory addresess (sets) so you need to sort the results

random_number_debug_counter = 0
def get_random_counter():
    global random_number_debug_counter
    return random_number_debug_counter

def report_random_counter():
    global random_number_debug_counter
    print('RND: %d' % random_number_debug_counter)

if False: # set to True to debug a replication problem
    print('NOTE: Running with wrapped random functions!')
    
    def update_random_counter():
        global random_number_debug_counter
        # Centralize so you set a break point here and condition on random_number_debug_counter
        random_number_debug_counter += 1
        
    def urandom_random():
        update_random_counter()
        return random.random()

    def urandom_shuffle(lst):
        update_random_counter()
        random.shuffle(lst) # no return

    def urandom_sample(lst,n):
        update_random_counter()
        return random.sample(lst,n)

    def urandom_choice(lst):
        update_random_counter()
        return random.choice(lst)
else:
    urandom_random  = random.random
    urandom_shuffle = random.shuffle
    urandom_sample  = random.sample
    urandom_choice  = random.choice

# See if we are being called from a debugger
def pdb_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    # return if we can tell some is debugging and someone appears to be debugging, else false
    # if they are debugging then gettrace is set to a function that is doing the debugging, else None
    # NOTE this only works if you have set a breakpoint!!
    return gettrace is not None and gettrace()


# Need a true function for multiprocessing support
# This function is run in each fork'd subprocess
def world_unfold(trial):
    global world
    filename = None
    if True and world.using_mp:
        # no reason to include os.getpid()
        filename = os.path.join(world.chronicler_directory,'trial_%d.log' % trial)
        # py3: Adding buffering=0 would ensure file is always flushed but that requires 'wb' but then you can't use print; not needed anyway
        sys.stdout = open(filename, "w")
        # Capture errors as well
        sys.stderr = sys.stdout

    world.unfold(trial)
    # TODO return (trial, filename,) and return value if any
    
# When we see bad behavior and need to debug it, assert its random_seed using -seed
# Random seeds are time stamps.
# NOTE: They don't yield the same unfolding behavior between different machines rendezvous (Ubuntu) and alban (MacOSX).
# NOTE also: replication using the same random seed is NOT ensured when running in the debugger
# presummably because the same underlying process is being reused and memory layout on subsequent runs
# is impacted by previous runs.  However, restarting pdb or running separate python instances or using
# the multiprocessing package starting parallel sessions (because they fork from pristine starts)
# using the same seed does work!
def execute(WHM_class):
    retval = 1

    # Force to be in UTC
    os.environ['TZ'] = 'UTC' 
    time.tzset()
    code_directory = os.path.abspath(os.path.dirname(sys.argv[0]))
    # let subclasses of WHM override handing of options
    options = argparse.ArgumentParser(prog=sys.argv[0],
                                      description='Run a world-historical model against a geography',
                                      conflict_handler='resolve')
    options.add_argument('geography_name',help='The geography to use', default=None) # This argument is required
    options.add_argument('-m',metavar='model',help='Model name to use',default=WHM_class.__name__)
    options.add_argument('-v', action='store_true', help='Verbose output ',default=False)    
    options.add_argument('-g',metavar='directory', help='Directory for geographies',default=code_directory)
    options.add_argument('-c',metavar='directory', help='Directory for code',default=code_directory)
    options.add_argument('-o',metavar='directory', help='Directory for output files',default=code_directory)
    options.add_argument('-t',metavar='tag',help='Output directory tag',default=None)
    # If you use -X it MUST be specified after geography_name or separated with another option
    # since it absorbs all tokens to the end or another option
    options.add_argument('-X',metavar='expr',action='append', nargs='*', help='Select experiment(s)',default=[])
    options.add_argument('--seed', type=int , help='Random seed for the trial') # 'default' is None, which is key to shuffling seeds
    options.add_argument('--n_trials', metavar='N', type=int , help='Number of trials',default=1) # if >1 will use multiprocessing unless disabled
    options.add_argument('--no_mp', action='store_true', help='Disable multiprocessing ',default=False)    
    options.add_argument('--profile', action='store_true', help='Gather timing statistics ',default=False) # eat this argument w/o complaint
    options.add_argument('--expand_supers', action='store_true', help='Expand supers code ',default=False)


    # This function times all the activities
    def trials():
        # initialize the world once, and unfold it n_trials
        global world
        # This makes the one requested WHM instance that all trials will use and parses the command line options
        # which sets/updates world.options to the result and a number of other flags
        world = WHM_class(options)
        # At this point we have the geography loaded and initial parameters are initialized, etc.

        if world.options.expand_supers: # have to wait to here to get all the modules loaded...
            expand_supers()
            return 1

        # NOTE: for some reason making movie frames per trial under multiprocessing kills the (master?) python process
        # but if MP is disabled it works fine
        # The bug might be just making figures since it requires a separate process to handle the display and that process is *not* forked...
        # To solve it requires changing the entire Chronicler plotting code to run in a separate process and ship the data you want via Pipe!
        # All the axes, etc. have to be in the separate process...
        # However, you typically want to make figures/movies only for one-off special applications so serial is fine
        world.using_mp = not world.options.no_mp
        if (world.options.no_mp or
            world.options.profile or
            world.n_trials == 1 or
            world.display_show_figures or
            pdb_debugging()):
            world.using_mp = False # avoid separate log files in world_unfold()

        if world.using_mp:
            # NOTE multiprocessing fails if machine is highly saturated or has to swap; use -no_mp
            import multiprocessing
            n_cpus = multiprocessing.cpu_count()
            print('Processing %d trials using %d processors' % (world.n_trials,n_cpus))
            pool = multiprocessing.Pool(n_cpus); # this is default pool size anyway
            # NOTE: if you run MP and have more trials than n_cpus, once the processes beyond n_cpus
            # run they *reuse* the forks in the pool and you'll find, for example, instance variables
            # (like western_cavalry_diffusion_limit, etc. in ADSM) already set from the last run
            # so make sure you re-initialize them
            results = pool.map(world_unfold,list(range(world.n_trials)))
            # TODO sort return values by trials (ensured?) and then 'cat' all the output log files to sys.stdout, then 'rm -f' them....
        else:
            # sequentially in this process
            print('Processing %d trials sequentially' % world.n_trials)
            for trial in range(world.n_trials):
                world_unfold(trial)
        return 1

    try:
        if "--profile" in sys.argv:
            pr = cProfile.Profile()
            pr.enable()
            retval = trials()
            pr.disable()
            # Report the stats
            stats = pstats.Stats(pr)
            stats.sort_stats('time', 'calls')
            stats.print_stats()
        else:
            retval = trials()
    except SystemExit:
        pass # be quiet...someone just wants to leave like parse_args()
    except: #  Exception
        # TODO handle KeyboardInterrupt?
        print("Unexpected error:", traceback.format_exc())
    return retval

def trial_basename(self,basename):
    tag = '_%d' % self.trial
    return basename % tag

skip_types = {np.ndarray, dict , set }
def print_scalar_members(self,print_values=True,header=None,footer=None,indent=""):
    # Given an instance of a class print its non-sequence and non-class instance members
    scalar_keys = []
    if print_values and header:
        print(header)
    keys = sorted(list(self.__dict__.keys()))
    for key in keys:
        value = self.__dict__[key]
        if isinstance(value,(int, float, complex, str)):
        # if type(value) is str or not (isinstance(value,collections.Sequence) or type(value) in skip_types):
            scalar_keys.append(key)
            if print_values:
                print('%s%s=%s' % (indent,key,value))
    if print_values and footer:
        print(footer)
    return scalar_keys

existing_subdirs = []
def driver(model_name,files,geographies,alt_name=None,generators=None):
    global existing_subdirs
    # Used to drive many runs and preserve the data and files
    # Parameters are set typically in <model_name>.py but you can add parameters on the command line
    import getpass # HACK to determine what machine we are on...
    amzn = getpass.getuser() == 'ubuntu' # else jsb on my local boxes
    n_trials = 4 # For Alban (2 cores, 4 hyperthreads)
    if amzn:
        n_trials = 1*8 # could scale by 2 AWS (8 cores, 16 hyperthreads)
    if alt_name is None:
        alt_name = model_name
        
    code_directory = os.path.abspath(os.path.dirname(sys.argv[0]))
    directory_tag = time.strftime("%m/%d/%Y %H:%M:%S UTC",time.gmtime(time.time()))
    options = argparse.ArgumentParser(prog=sys.argv[0])
    options.add_argument('-t',metavar='tag',help='Output directory tag',default=directory_tag)
    options.add_argument('-g',metavar='generator',help='Which variable generator',default=None)
    options.add_argument('-z',action='store_true', help='Compress generator results directories ',default=False)    
    options.add_argument('-m',metavar='model',help='Model name to pass to code',default=alt_name)
    options.add_argument('-o',metavar='directory', help='Directory for output files',default=code_directory)
    options.add_argument('--n_trials', metavar='N', type=int , help='Number of trials',default=n_trials)
    # separate arguments to pass to program using -- <model_args>
    options.add_argument('model_args', nargs=argparse.REMAINDER)
    options = options.parse_args()

    directory_tag = options.t
    alt_name = options.m
    if len(options.model_args) > 0:
        options.model_args = options.model_args[1:] # strip '--'

    files.append("%s.py" % model_name)
    # Common support files
    files.extend([sys.argv[0]]) # the driver() calling code
    files.extend(["Utils.py","Constants.py","Geography.py"])
    files.extend(["WHM.py","Chronicler.py","ComparativeHx.py"])
    files.extend(["Population.py","PopChronicler.py"]) # for tracking population
    # files.extend(["create_geography.py","create_arena.py"])
    # support hill climb overrides as a way of asserting parameters
    ff = '%s_fixed.py' % alt_name
    if os.path.exists(ff):
        files.extend([ff])
    vf = '%s_vary.py' % alt_name
    if os.path.exists(vf):
        files.extend([vf])
    
    # Add the geographies
    geo_pkl = ["%s.pkl" % geography for geography in geographies]
    files.extend(geo_pkl)

    # Create a <model_name>_<date> subdirectory
    # Copy the list of files to it
    # then cd to it and execute the given model code against all the geographies, preserving the run output
    # Use UTC to match to figure tags
    print("Copying code from %s" % code_directory)
    if options.g:
        basename = ensure_basename("%s_%s_%s" % (model_name,options.g,directory_tag))
    else:
        basename = ensure_basename("%s_%s" % (model_name,directory_tag))
    work_directory = os.path.join(os.path.abspath(options.o), basename)
    try:
        os.makedirs(work_directory)
    except:
        print("Unable to make work directory %s" %  work_directory)
        return
    else:
        print("Saving code and work to %s" %  work_directory)

    for f in files:
        try:
            copied_file = os.path.join(work_directory,f)
            shutil.copyfile(f,copied_file)
            # make read-only so we aren't tempted to edit it
            os.chmod(copied_file,S_IRUSR|S_IXUSR|S_IRGRP|S_IXGRP) # read/execute only for usr and group
        except:
            print("Unable to copy %s to work directory %s" %  (f,work_directory))
            return
    
    os.chdir(work_directory)
    code_name = os.path.join(work_directory,model_name)

    # Each run will generate a subdir but in general we don't know its name (since model_args might include -X etc.)
    # so we maintain a list of known subdirs and look, after each run the new subdir and deal with it for copying and -z
    existing_subdirs = [] # reset
    def find_new_subdirs(permit_empty=False):
        global existing_subdirs
        current_subdirs = [f.path for f in os.scandir('.') if f.is_dir()]
        # this works even if some of the existing subdirs are deleted since last we looked
        # also, this returns a list, which could be empty or have several
        new_subdirs = setdiff(current_subdirs,existing_subdirs)
        existing_subdirs = current_subdirs
        if len(new_subdirs) == 0:
            if not permit_empty:
                raise RuntimeError('Unable to find results subdirectory!')
            return None
        if len(new_subdirs) > 1:
            raise RuntimeError('Found multiple results subdirectories! %s' % new_subdirs)
        subdir = new_subdirs[0]
        return subdir[2:] # strip './'
    find_new_subdirs(permit_empty=True) # initialize existing_subdirs
    existing_subdirs.append('./__pycache__'); # py3 will create this when we run code, which we ignore
    try:
        for geography in geographies:
            print("Running %s in %s at %s" % (model_name, geography, time.strftime("%m/%d/%Y %H:%M:%S",time.localtime(time.time()))))
            tag = "%s_%s" % (alt_name,geography)
            logfile = ensure_basename("log_%s" % tag)
            cmdline = "python3 %s.py -m %s --n_trials %d %s %s > %s"
            cmdline = cmdline % (code_name, alt_name, options.n_trials, geography,' '.join(options.model_args),logfile)
            if options.g is not None and generators is not None:
                for id in variable_generator(generators,options.g,vf):
                    run_cmd_shell(cmdline)
                    results_subdir = find_new_subdirs()
                    # if we get here results_subdir exists
                    logfile = ensure_basename("log_%s" % tag)
                    shutil.move(logfile,"%s/%s" % (results_subdir,logfile)) # move logfile to subdir
                    shutil.move(vf,"%s/%s" % (results_subdir,vf)) # move the variable file to subdir
                    tag_id = '%s_%d' % (results_subdir, id)
                    shutil.move(results_subdir,tag_id) # rename the directory to indicate which generator set was used
                    if options.z:
                        run_cmd_shell('tar cfz %s.tgz %s' % (tag_id,tag_id))
                        shutil.rmtree(tag_id,ignore_errors=True)
                        find_new_subdirs(permit_empty=True) # update existing_subdirs
            else:
                run_cmd_shell(cmdline)
                if options.z:
                    results_subdir = find_new_subdirs()
                    run_cmd_shell('tar cfz %s.tgz %s' % (results_subdir,results_subdir))
                    shutil.rmtree(results_subdir,ignore_errors=True)
                    find_new_subdirs(permit_empty=True) # update existing_subdirs
    except RuntimeError as exception:
        import traceback
        print("Unexpected error:", traceback.format_exc())

    if options.z:
        run_cmd_shell('tar cfz geographies.tgz %s' % ' '.join(geo_pkl))

    # Do this cleanup separately and last so the dates sort well in the Finder...what a hack
    # BUG: in the ..X.py wrapper we sometimes set name to something else like 'Historical'
    # And hence the subdir gets named not ...X_OldWorld but Historical_OldWorld
    # Thus the code below fails because we don't think the path exists...
    for geography in geographies:
        if os.path.exists(tag):
            if False:
                logfile = ensure_basename("log_%s" % tag)
                shutil.move(logfile,"%s/%s" % (tag,logfile)) # move logfile to subdir
            data_png = "%s/Data.png" % tag
            if os.path.exists(data_png):
                os.symlink(data_png,"%s.png" % tag)
    run_cmd_shell("rm -rf *.pyc __pycache__") # Get rid of compiled files
    os.chdir("..") 
    print("Finished running in %s at %s" % (work_directory, time.strftime("%m/%d/%Y %H:%M:%S",time.localtime(time.time()))))

class variable_generator(object):
    def __init__(self,generators,which,filename):
        self.which = which
        self.variable_ranges = generators[which]
        max_index = None
        for var,values in list(self.variable_ranges.items()):
            if var == 'CONSTANTS':
                continue
            len_values = len(values)
            if max_index is None:
                max_index = len_values
            elif len_values is not max_index:
                raise RuntimeError('Mismatched number of values (%d vs. %d) for %s in %s generator' % (len_values,max_index,var,which))
        self.max_index = max_index
        self.filename = filename
        self.index = 0

    def __repr__(self):
        return "<var_gen %s %s %d>" % (self.which, list(self.variable_ranges.keys()), self.index)

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return next(self)
 
    def __next__(self):
        index = self.index
        if index < self.max_index:
            fid = open(self.filename,'w')
            gen_id = '%s[%d]' % (self.which,index)
            print(' Running %s:' % gen_id)
            fid.write('# %s\n' % gen_id)
            for var,values in list(self.variable_ranges.items()):
                if var == 'CONSTANTS':
                    continue
                assignment = '%s = %s' % (var,values[index])
                print('  %s' % assignment) # indented
                fid.write('self.%s\n' % assignment)
            try:
                # Write constants, if any, last so they can access values asserted above
                for assignment in self.variable_ranges['CONSTANTS']:
                    fid.write('self.%s\n' % assignment)
            except KeyError:
                pass
            fid.close()
            self.index += 1
            return index
        else:
            raise StopIteration()
    
class polity_flag(object):
    # A generator (limit < 0) or a singleton emitter
    # pass the shuffle function along in case we create an instance of a different random number generator
    # colors: avoid black and green since those are our background
    # markers: avoid '.' since those our used for regions
    def __init__(self,limit=-1,shuffle=urandom_shuffle,colors='rcmb',markers='+odx^vs*<>ph'):
        self.index = 0
        self.limit = limit
        # Expand this to use better color names from matplot lib
        # each flag is a tuple?
        flags = []
        for c in colors: 
            for m in markers: 
                flags.append({'marker':m, 'color':c})
        shuffle(flags) # shuffle in place, returns None, done once per instsance
        self.flags = flags # set shuffled list

    def __repr__(self):
        return "<Flag %s %s>" % ("generator" if self.limit < 0 else "emitter", self.flags[self.index]) # show next flag

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return next(self)
 
    def __next__(self):
        flag = self.flags[self.index]
        # In the iterator case we count down to zero
        self.limit -= 1
        if self.limit == 0:
            raise StopIteration()
        self.index += 1
        if self.index >= len(self.flags):
            self.index = 0
        return flag

def run_cmd_shell(cmd):
    """Run cmd in a subshell
    """
    # TODO wrap in a try block, turn on signal.signal(signal.SIGHUP,sighup_handler),
    # which throws a RuntimeError, which os.kills the subprocess before exiting itself.
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    # sts >> 8 to get return code
    pid,sts = os.waitpid(p.pid, 0)
    return (sts, p.stdout)

def add_unique(list,elt):
    """Add an element uniquely to a list
    """
    # Since we can't use set(), which uses memory addresses as hashes
    # for any creation or order-sensitive operation in models
    # instead we use lists and rather than foo = set(); foo.add(x)
    # we use foo = []; add_unique(foo,x)
    if elt not in list:
        list.append(elt)
    return list

def unique(s):
    """Return a list of the elements in s, but without duplicates.
    """
    if use_set_versions:
        return list(set(s))
    else:
        unique_list = []
        unique_set = set()
        # preserve order of s
        for elt in s:
            if elt not in unique_set:
                unique_set.add(elt)
                unique_list.append(elt)
        return unique_list

def intersect(list1,list2):
    '''Return the intersection of two lists
    Inputs:
    list1,list2 - the lists

    Returns:
    their set intersection
    
    Raises:
    None
    # NOTE: This uses memory addresses so the results are not-canonical between runs!!
    '''
    # np.intersect1d uses explicit sort which calls __lt__ which you would need on objects (Territories)
    # return np.intersect1d(np.array(list1,object),np.array(list2,object)).tolist()
    return list(set(list1).intersection(set(list2)))

def union(list1,list2):
    '''Return the union of two lists
    Inputs:
    list1,list2 - the lists

    Returns:
    their set union
    
    Raises:
    None
    # NOTE: This uses memory addresses so the results are not-canonical between runs!!
    '''
    # ensures unique()
    return np.union1d(np.array(list1,object),np.array(list2,object)).tolist()

def setdiff(list1,list2):
    '''Return the set difference of two lists
    Inputs:
    list1,list2 - the lists

    Returns:
    their set difference
    
    Raises:
    None
    # NOTE: This uses memory addresses so the results are not-canonical between runs!!
    '''
    return np.setdiff1d(np.array(list1,object),np.array(list2,object)).tolist()

def index_i(list1,list_i):
    '''Return a list of elements of list1 selected by indicices in list_i
    E.g., under numpy list1[list_i] but handles lists and returns lists...

    Inputs:
    list1 - the list or array
    list_i - the list or array of indices

    Returns:
    a list of elements
    
    Raises:
    None
    '''
    return np.array(list1)[list_i].tolist()

def succinct_elts(elts,matlab_offset=1):
    '''Return a string of numeric elts, succinctly showing runs of consecutive values, if any
    Inputs:
    elts - a set of integers
    matlab_offset - offset to use if these are NOT indices

    Returns:
    selts - a succinct string
    
    Raises:
    None
    '''
    elts = np.sort(unique(elts))
    elts = elts + matlab_offset
    selts = ""
    prefix = ""
    num_elts = len(elts)
    if (num_elts):
        diff_elts = np.diff(elts)
        breaks_i_v = [i for i in range(len(diff_elts)) if diff_elts[i] > 1]
        breaks_i_v.append(len(elts)-1) # add the final point
        last_i = 0
        for break_i in breaks_i_v:
            nelts =  elts[break_i] - elts[last_i]
            if (nelts == 0):
                selts = "%s%s%d" % (selts,prefix,elts[break_i])
            elif (nelts == 1):
                selts = "%s%s%d %d" % (selts,prefix,elts[last_i],elts[break_i])
            else:
                selts = "%s%s%d:%d" % (selts,prefix,elts[last_i],elts[break_i])
            last_i = break_i+1
            prefix = " "
    return selts

def get_key(dict_d,key,default=None):
    '''Looks for key in dict_d, returns that value otherwise returns default (like getattr but for dicts)'''
    try:
        return dict_d[key]
    except KeyError:
        return default
    
def Oxford_comma(strings,connector='and'):
    '''Returns a string of string elements, in given order, serial comma-separated according to the Oxford rule.
    '''
    n = len(strings)
    if (n == 0):
        return ''
    elif (n == 1):
        return strings[0]
    elif (n == 2):
        return '%s %s %s' % (strings[0], connector, strings[1])
    else:
        string = ''
        for i in range(n):
            string = '%s%s, ' % (string,strings[i])
            if (i == n-2):
                string = '%s%s %s' % (string,connector,strings[i+1])
                break
        return string
        
def ensure_basename(basename):
    '''Returns basename with problematic filename characters replaced
    
    Inputs:
    basename - string

    Returns:
    basename possibly modified
    
    Raises:
    None
    '''
    basename = basename.replace(' ', '_')
    basename = basename.replace(',', '_')
    basename = basename.replace('/', '_')
    basename = basename.replace(':', '_') # makedirs() replaces : with /!
    basename = basename.replace('&', '_')
    return basename


def ask_yes_no(prompt,default=True):
    default_tag = 'Yn' if default else 'Ny'
    # Don't use input(), which expects an expression to evaluate
    answer = input('%s (%s) ' % (prompt,default_tag))
    if len(answer):
        answer = answer.lower()
        return (answer[0] == 'y')
    else:
        return default

def pretty_year(year,padded=True):
    ce = ' CE' if padded else 'CE'
    return '%d%s' % (abs(year), 'BCE' if year < 0 else ce)

# DEAD
# A mixin that makes the instances of different classes reusable
# Make sure it is before 'object' in the supers list when mixing so these __new__ and __del__ methods are seen first
class Reusable(object):
    # if we mix this into several classes they will be mixed but saved by their class type
    # BUG mixing this on WHM:Order yields TypeError: object() takes no parameters
    recycled_instances = {} # <class> -> [<instance> ...]
    def __new__(cls,*arguments,**kwargs):
        try:
            instances = cls.recycled_instances[cls]
        except KeyError:
            instances = []
            cls.recycled_instances[cls] = instances

        try:
            instance = instances.pop()
        except IndexError:
            instance = super().__new__(cls,*arguments,**kwargs)
        # __init__ called on the instance next??
        return instance

    def __init__(self,*arguments,**kwargs):
        
    def __del__(self):
        # GC thinks no one is using this
        # but we beg to differ, so we can reuse this instance
        # We know there should be an entry in the hash for this class instance
        self.recycled_instances[type(self)].append(self)


# NOTE: For the following functions can use P0, P, and K as fractions of 1
# Units of time depend on the units of beta;  typically beta is a percent per year (0.02 == 2% net growth rate)
# so growth times will be in years.

# given a starting population P0, a carrying capacity K, a growth rate beta (typically between 0 and 1), and a time
# what is the ending population?
def logistic_P(P0,K,beta,time):
	return (K*P0*np.exp(beta*time))/(K + P0*(np.exp(beta*time) - 1.0))

# given an ending population P, a carrying capacity K, a growth rate beta (typically between 0 and 1), and a time
# what was the starting population?
def logistic_P0(P,K,beta,time):
    return (P*K)/(K*np.exp(beta*time) - P*(np.exp(beta*time) - 1))

# given a starting population P0, and ending population P, a carrying capacity K, a growth rate beta (between 0 and 1), 
# what is the time to for P0 to reach P?
def logistic_t(P0,P,K,beta):
	return np.log((P/P0)*((K - P0)/(K - P)))/beta

# given a starting population P0, and ending population P, a carrying capacity K, and a time
# what is the net birth rate (beta) required?
# NOTE: You can use K = 1 then P0,P as relative fractions of K to compute rise times
def logistic_beta(P0,P,K,time):
	return np.log((P/P0)*((K - P0)/(K - P)))/time

def millions(value):
    # for plotting, return value as a string xxK, xxM or xxB for kilo, mega, and giga values
    if value < 1e3:
        return "%.1f" % value
    elif value < 1e6:
        return "%.1fK" % (value/1e3)
    elif value < 1e9:
        return "%.1fM" % (value/1e6)
    else:
        return "%.1fB" % (value/1e9)

# simple code to expand supers calls into a single file
import inspect
def extract_between(raw_line,start_tag,end_tag,startswith=False,return_remainder=False):
    if startswith:
        start_i = raw_line.startswith(start_tag);
    else:
        start_i = raw_line.find(start_tag);
    if start_i >= 0:
        raw_line = raw_line[start_i + len(start_tag):]
        end_i = raw_line.find(end_tag);
        if end_i >= 0:
            if return_remainder:
                return (raw_line[0:end_i],raw_line[end_i+1:])
            else:
                return raw_line[0:end_i]
    if return_remainder:
        return (None,raw_line)
    else:
        return None

import inspect
def expand_supers():
    # collect all our classes by module in the code directory
    code_dir = world.options.c
    expand_dir = os.path.join(os.path.abspath(code_dir),'expanded_src')
    shutil.rmtree(expand_dir,ignore_errors=True)
    try:
        os.makedirs(expand_dir)
    except:
        raise RuntimeError("Unable to make expanded code directory %s" %  expand_dir)

    class_mro = {}
    mod_names = []
    for mname,mod in sys.modules.items():
        mod_location = getattr(mod,'__file__',None)
        if mod_location:
            mod_dir = os.path.abspath(os.path.dirname(mod_location))
            if mod_dir == code_dir:
                mod_names.append(mname)
                for name,cls in inspect.getmembers(mod, inspect.isclass):
                    class_mro[name] = list(cls.__mro__)
    mod_names = unique(mod_names)

    print('Module names:')
    pprint.pprint(mod_names)
    print('Class mro:')
    pprint.pprint(class_mro)

    out_file = None
    top_module = None
    def expand_super(efn,mro,super_line=None):
        cmro = mro.copy()
        # Find the first class on (remaining) mro that contains the efn
        while cmro:
            cls = cmro.pop(0)
            if cls == object:
                continue
            try:
                cls.__dict__[efn]
            except KeyError:
                continue # not defined on cls

            module = cls.__module__
            eclass = cls.__name__
            # Don't expand code that will be expanded in the same module (eventually)
            if super_line and module == top_module:
                out_file.write('## super calls %s:%s(...)\n' % (eclass,efn))
                out_file.write(super_line)
                break
            mod_filename = sys.modules[module].__file__
            try:
                mod_file = open(mod_filename, "r")
            except IOError:
                print("ERROR: Could not open %s for reading." %  mod_filename)
                return

            current_class = None
            current_fn = None
            found_code = False
            tag = '%s:%s:%s\n' % (os.path.split(mod_filename)[1],eclass,efn)
            for raw_line in mod_file:
                line = raw_line.lstrip()
                if line.find('class ') == 0:
                    current_class = extract_between(line,'class ','(')
                if line.find('def ') == 0:
                    current_fn = extract_between(line,'def ','(')
                if current_class == eclass and current_fn == efn:
                    if not found_code:
                        out_file.write('##s %s' % tag)
                        found_code = True
                        continue # skip echoing the def line
                    # In fact we need to search mro for the first hit of efn and expand it
                    # plus send the mro of that class in case super() is called by the next method
                    if line.find('super(') == 0:
                        # out_file.write('##> %s' % tag)
                        expand_super(efn,cmro)
                        # indicate we have returned from super()
                        # so we know where next code fragments come from
                        out_file.write('##< %s\n' % tag) 
                    else:
                        out_file.write(raw_line)
                if found_code and (current_class != eclass or current_fn != efn):
                    out_file.write('##e %s' % tag)
                    break # done expanding
            mod_file.close()
            break


    for mod in mod_names:
        top_module = mod
        mod_filename = sys.modules[mod].__file__
        try:
            mod_file = open(mod_filename, "r")
        except IOError:
            print("ERROR: Could not open %s for reading." %  mod_filename)
            continue
        out_basename = os.path.splitext(os.path.split(mod_filename)[1])[0]
        out_filename = os.path.join(expand_dir, out_basename + '_expanded.py')
        try:
            out_file = open(out_filename,'w')
        except IOError:
            print("ERROR: Could not open %s for writing." %  out_filename)
            continue

        current_class = None
        current_fn = None
        expanded = False
        for raw_line in mod_file:
            line = raw_line.lstrip()
            if line.find('class ') == 0:
                current_class = extract_between(line,'class ','(')
                if current_class is None:
                    current_class = extract_between(line,'class ',':') # no inheritance
                if True: # list all the other methods available via inheritance
                    out_file.write(raw_line)
                    try:
                        for name_func in inspect.getmembers(class_mro[current_class][0],inspect.isfunction):
                            # these are in alphabetic order
                            name,func = name_func
                            if func.__module__ != mod:
                                expanded = True
                                out_file.write('##f %s:%s(...)\n' % (func.__module__,name))
                    except KeyError:
                        current_class
                        pass # current_class 
                    continue
            if line.find('def ') == 0:
                current_fn = extract_between(line,'def ','(')

            if line.find('super(') == 0:
                expanded = True
                expand_super(current_fn,class_mro[current_class][1:],raw_line)
            else:
                out_file.write(raw_line)
            
        # close mod file
        mod_file.close()
        if not expanded:
            os.remove(out_filename)
        
