
import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.dirname(os.path.realpath(inspect.getfile(lambda: None))))
print('Main dir:', __MAINDIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)

import copy, re
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ray import tune
from ray.tune import Analysis
from ray.tune.utils import flatten_dict

#data filtering
from scipy.signal import medfilt

from common.util import setup_logger, make_new_dir

EXPERIMENT_STATE_FILE = 'experiment_state*.json'


class ExperimentPlotter():
    """Plotter class for generating graphs from a CSV file (of Ray.Tune).
    """
    headerStruct = None
    CATEGORY_COL = ''
    VALUE_COL = ''

    def __init__(self, 
                 output_folder, 
                 csv_path, 
                 *, 
                 is_melted=False, 
                 plot_style='darkgrid',
                 logger=None):
        self.logger = logger or setup_logger(__name__, verbose=True)
        self.__output_dir = None
        self.output_dir = output_folder

        self.csv_path = csv_path
        assert Path(self.csv_path).is_file(), "CSV file does not exist!"

        self.df = pd.read_csv(self.csv_path, index_col=False)


        #the plot style, one of darkgrid, whitegrid, dark, white, ticks
        self.plot_style = plot_style

        self._convert_df(is_melted)

        self.show_plots = False
        self.save_plots = False

    @property
    def output_dir(self):
        return self.__output_dir
    @output_dir.setter
    def output_dir(self, path):
        if not Path(path).is_dir():
            self.logger.debug(f'Creating output directory: {path}')
            os.makedirs(path)
        self.__output_dir = path


    @contextmanager
    def context(self, **kwargs):
        '''Context manager for setting arbitrary object members to
        a desired value temporarily.
        '''
        tmp_store = {}
        try:
            for k,v in kwargs.items():
                if hasattr(self, k):
                    tmp_store[k] = getattr(self, k)
                    setattr(self, k , v)
                else:
                    self.logger.warning(f'Context Manager: attribute {k} not a '+
                                        f'member of {self.__class__.__name__}. Ignoring')
            yield
        finally:
            for k,v in tmp_store.items():
                setattr(self, k , v)

    def _convert_df(self, is_melted=False):
        """This must be implemented in sub-classes depending on 
        the structure of the experiment output.
        """
        raise NotImplementedError

    def save_melted_df(self, path):
        # assert Path(path).exists(), \
        #     'Output file already exists, please use a different path!'
        self.logger.debug(f'Saving long dataframe to: {path}')
        self.df.to_csv(path, index=False, na_rep='nan')


    def _validate_colname(self, name):
        if name in self.headerStruct.keys():# or name in self.df.columns:
            return name, name
        else:
            for k,v in self.headerStruct.items():
                if name in v.keys():
                    return name, k
                elif name in v.values():
                    return list(v.keys())[list(v.values()).index(name)], k

        self.logger.warning('Value name `{}` for plotting is '.format(name) + \
                            'neither in the header mapping keys/values nor in '
                            'the loaded data frame!')
        return False, ''


    def _to_column_list(self, value):
        _value = value
        if isinstance(value, str):
            value, category = self._validate_colname(value)
            assert value, "{} not in the headerStruct or Dataframe".format(_value)
            if value == category:
                value = list(self.headerStruct.get(category,{}).keys()) or \
                        list(self.df[category].unique())
            else:
                value = [value]
            categories = [category] * len(value)

        elif isinstance(value, (list,tuple)):
            categories = []
            vals = []
            for v in value:
                assert isinstance(v, str), 'Values in the list must be strings!'
                names, cat = self._to_column_list(v)
                vals += names
                categories += cat
        
        else:
            raise ValueError('The value to plot must be a single string or '
                             'list/tuple, got {}!'.format(type(_value)))
        
        return value, categories
    
    def _get_sub_df(self, category_vals):
        return self.df[self.df[self.CATEGORY_COL].isin(category_vals)]


    def plot_data(self, x_vals, y_vals, plot_type='single', plot_style="line", **options):
        """Call function like e.g.
            p = TuneExperimentPlotter(...)
            p.plot_data('Index', 'Rewards', **options)
        to produces `len(p.headerStruct['Index'])` multi- or 
        `len(...['Index']) * len(...['Rewards'])` single-plots.

        Call function like 
            p.plot_data('Timesteps', 'Rewards', **options)
        to produce one multi- or `len(p.headerStruct['Rewards'])` 
        single-plots with 'Timesteps' as the x-axis.
        """
        handles = []
        x_vals, xCats = self._to_column_list(x_vals)
        y_vals, yCats = self._to_column_list(y_vals)

        self.logger.debug(f'Creating {plot_type} plot for '+ \
                         f'{y_vals} values on {x_vals} x-axes...')
        
        if plot_type == 'multi':
            sns.set(style="ticks")
        else:
            sns.set(style=self.plot_style)
        
        if not plot_type == 'adjacent':
            if plot_style == "line":
                options.update(kind="line")
            elif plot_style == "line_sd":
                options.update({'kind': "line", 'ci': "sd"})
            elif plot_style == "scatter":
                options.update(kind="scatter")
            else:
                raise ValueError('Unknown plot type ({}) to use for data!'.format(plot_style))

        _options = copy.deepcopy(options)
        #for each given x axes, draw all the value columns
        for xName, xCat in zip(x_vals, xCats):
            options = copy.deepcopy(_options)
            if plot_type in ['multi','adjacent']:
                
                df = self._get_sub_df(y_vals).astype({self.VALUE_COL: float})  #filter for wanted values
                if plot_type == 'multi':
                    g = sns.relplot(x=xName, y=self.VALUE_COL, hue=self.CATEGORY_COL, 
                                    data=df, 
                                    **options)

                elif plot_type == 'adjacent':
                    from math import sqrt, ceil
                    col_wrap = options.pop('col_wrap', ceil(sqrt(len(y_vals))))
                    col_wrap = col_wrap if len(y_vals) > 3 else len(y_vals)
                    color = options.pop('color', "#bb3f3f")     # #bb3f3f : wine red
                    height = options.pop('height', 3)
                    aspect = options.pop('aspect', 1)

                    g = sns.FacetGrid(df, col=self.CATEGORY_COL, col_wrap=col_wrap, 
                                      height=height, aspect=aspect, sharey=False)
                    g.map(sns.lineplot, xName, self.VALUE_COL, color=color, ci=None, 
                          estimator=options.pop('estimator',None), 
                          **options)
                    g.set_titles("{col_name}")

                g.set(ylabel=None)
                xName_p = xName.replace(' ','').replace(']','').replace('[','_')
                f_name = f'Plot_{plot_type}_{xName_p}'
                g.fig.canvas.set_window_title(f_name)


                if self.save_plots:
                    out_path = Path(self.output_dir) / f'{f_name}.png'
                    g.savefig(str(out_path)) #FacetGrid has savefig()

                handles.append(g)

            elif plot_type == 'single':
                for yName, yCat in zip(y_vals, yCats):
                    df = self._get_sub_df([yName])
                    if isinstance(df.iloc[0][self.VALUE_COL], (list,tuple)):  #if we deal with list data
                        df = df.explode(self.VALUE_COL).dropna()
                    df = df.astype({self.VALUE_COL: float})
                    
                    g = sns.relplot(x=xName, y=self.VALUE_COL, data=df, **options)
                    g.set(ylabel=yName)
                    xName_p = xName.replace(' ','').replace(']','').replace('[','_')
                    yName_p = yName.replace(' ','').replace(']','').replace('[','_')
                    f_name = f'Plot_{plot_type}_{xName_p}_{yName_p}'
                    g.fig.canvas.set_window_title(f_name)

                    if self.save_plots:
                        out_path = Path(self.output_dir) / f'{f_name}.png'
                        g.savefig(str(out_path)) #FacetGrid has savefig()
                    
                    handles.append(g)
            
            else:
                raise ValueError(f'Unknown plot_type {plot_type}!')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
        
        return handles


class TuneExperimentPlotter(ExperimentPlotter):

    # Header structure of desired plot names (keys) and 
    # passed csv headers (values)
    # Units in the key names can be given in '[]'
    headerStruct = {
        'Index':{
            'Timesteps': 'timesteps_total',
            'Episodes': 'episodes_total',
            'Training Iterations': 'training_iteration',
            'Wall Time [s]': 'time_total_s'},
        'Rewards':{
            'Mean Reward': 'episode_reward_mean',
            'Max Reward': 'episode_reward_max',
            'Min Reward': 'episode_reward_min'},
        'Losses':{
            'Total Loss': 'info/learner/default_policy/total_loss',
            'Policy Loss': 'info/learner/default_policy/policy_loss',
            'Value Estimation Loss': 'info/learner/default_policy/vf_loss'},
        'Episode Stats':{
            'Episode Mean Length': 'episode_len_mean',
            'Completed Episodes': 'episodes_this_iter'},
        'Timings':{
            'Sample Time [ms]': 'timers/sample_time_ms',
            'Learn Time [ms]': 'timers/learn_time_ms',
            'Policy Update Time [ms]': 'timers/update_time_ms'},
        'Hyperparams':{
            'Learning Rate': 'info/learner/default_policy/cur_lr',
            'Explained Variance': 'info/learner/default_policy/vf_explained_var',
            'KL Divergence': 'info/learner/default_policy/kl',
            'KL Coefficient': 'info/learner/default_policy/cur_kl_coeff',
            'Entropy': 'info/learner/default_policy/entropy',
            'Entropy Coefficient': 'info/learner/default_policy/entropy_coeff'},
        'HW Utilization':{
            'CPU Utilization [perc]': 'perf/cpu_util_percent',
            'GPU Utilization [perc]': 'perf/gpu_util_percent0',
            'RAM Utilization [perc]': 'perf/ram_util_percent',
            'VRAM Utilization [perc]': 'perf/vram_util_percent0'},
        'Hist Data':{
            'Rewards per Episode': 'hist_stats/episode_reward',
            'Episode Lengths': 'hist_stats/episode_lengths'},
        }
    
    CATEGORY_COL = 'CATEGORY'
    VALUE_COL = 'VALUES'
    
    def __init__(self, 
                 output_folder, 
                 progressCSV_path, 
                 *, 
                 is_melted=False, 
                 plot_style='darkgrid',
                 logger=None):
        from ast import literal_eval
        super().__init__(output_folder, progressCSV_path, 
                         is_melted=is_melted, plot_style=plot_style,
                         logger=logger)
        #assume that hist data is in serialized list format and needs 
        # to be converted back to a proper python list
        hist_cat_names, _ = self._to_column_list('Hist Data')
        self.df[self.VALUE_COL].update(
            self._get_sub_df(hist_cat_names)[self.VALUE_COL].apply(literal_eval)
            #note: saver literal_eval is slower than unsave eval()
        )


    def _convert_df(self, is_melted=False):
        """Convert self.df to a dataframe with self.headerStruct
        keys as column names and values as categorical values.
        (Necessary for Seaborn)
        The new DF will have columns:
            - self.headerStruct['Index'].keys()
            - self.headerStruct.keys()
            - self.headerStruct.keys() + '_values'
        """
        if is_melted:   #if the imported CSV is already in the right format
            return
        self.logger.debug('Unpivoting dataframe to categorical long format...')

        #replace all invalid values with np.nan
        # NaN is used as identifier to invalid values for the plotting functions
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        indexes = list(self.headerStruct['Index'].values())
        var_names = [vv for k,v in self.headerStruct.items() for vv in v.values() if k != 'Index' and vv in list(self.df.columns)]
        self.df = self.df.melt(id_vars=indexes, 
                               value_vars=var_names,
                               var_name=self.CATEGORY_COL,
                               value_name=self.VALUE_COL)

        # rename the categorical values according to headerStruct
        name_mappings = {v:k for cat,maps in self.headerStruct.items() for k,v in maps.items() if cat!='Index' and v in list(self.df.columns)}
        self.df[self.CATEGORY_COL].replace(name_mappings, inplace=True)
        # rename the id column names according to headerStruct
        self.df.rename(columns={v:k for k,v in self.headerStruct['Index'].items() if v in list(self.df.columns)}, errors="raise", inplace=True)


    def plot_hists(self, x=None, hist_cats=None, plot_type=True, **options):
        self.logger.debug('Creating plots for hist data...')
        handles = []

        if not x:
            x = 'Index'
        if not hist_cats:
            hist_cats = 'Hist Data'
        
        x_vals, xCats = self._to_column_list(x)
        y_vals, yCats = self._to_column_list(hist_cats)

        sns.set(style=self.plot_style)

        # only use categories that were given
        for yName in y_vals:
            df = self._get_sub_df([yName])
            df = df.explode(self.VALUE_COL).dropna().astype({self.VALUE_COL: float})  #expand the list entries to own rows
            for xName in x_vals:
                g = sns.relplot(x=xName, y=self.VALUE_COL, kind="line", ci="sd", data=df, **options)
                g.set(ylabel=yName)
                xName_p = xName.replace(' ','').replace(']','').replace('[','_')
                yName_p = yName.replace(' ','').replace(']','').replace('[','_')
                f_name = f'Histo_{xName_p}_{yName_p}'
                g.fig.canvas.set_window_title(f_name)

                if self.save_plots:
                    out_path = Path(self.output_dir) / f'{f_name}.png'
                    g.savefig(str(out_path)) #relplot returns a FacetGrid
                
                handles.append(g)


        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

        return handles

    
    def plot_hypers(self, x=None, hp_cats=None, **options):
        """Options:
            - plot_dist: whether to plot the distribution view of the HPs
            - col_wrap: numper of subplots in one row
            - color: color of the plot lines (e.g. "#bb3f3f", 'g')
            - height: height of each subplot
            - aspect: aspect ratio from height to width
        """
        self.logger.debug('Creating plots for hyperparameters...')
        from math import sqrt, ceil
        handles = []

        if not x:
            x = 'Index'
        if not hp_cats:
            hp_cats = 'Hyperparams'

        x_vals, xCats = self._to_column_list(x)
        y_vals, yCats = self._to_column_list(hp_cats)
 
        sns.set(style='ticks')

        #plot the HPs in a facet grid
        col_wrap = options.pop('col_wrap', ceil(sqrt(len(y_vals))))
        col_wrap = col_wrap if len(y_vals) > 3 else len(y_vals)
        color = options.pop('color', "#bb3f3f")     # #bb3f3f : wine red
        height = options.pop('height', 3)
        aspect = options.pop('aspect', 1)

        # only use hp categories that were given
        df =self._get_sub_df(y_vals).astype({self.VALUE_COL: float})
        #ceate the same plot(s) for each x-value given
        for xName in x_vals:

            g = sns.FacetGrid(df, col=self.CATEGORY_COL, col_wrap=col_wrap, height=height, aspect=aspect, sharey=False)
            g.map(sns.lineplot, xName, self.VALUE_COL, color=color, ci=None, estimator=None)
            g.set_titles("{col_name}")
            g.set(ylabel=None)
            xName_p = xName.replace(' ','').replace(']','').replace('[','_')
            f_name = 'Hyperparams_{}'.format(xName_p)
            g.fig.canvas.set_window_title(f_name)
            
            if self.save_plots:
                out_path = Path(self.output_dir) / f'{f_name}.png'
                g.savefig(str(out_path)) #FacetGrid has savefig()

            handles.append(g)
        
        # distribution is independent of the x value, i.e. only do once
        if options.pop('plot_dist', False):
            #check if some HP are singular
            dist_hps = []
            for cat in df[self.CATEGORY_COL].unique():
                tmp_values = df.loc[df[self.CATEGORY_COL]==cat,self.VALUE_COL].to_list()
                if not tmp_values:  #list empty
                    self.logger.debug(f'HP dist plot: no values for {cat}. Ignoring')
                elif tmp_values.count(tmp_values[0]) == len(tmp_values):
                    self.logger.debug(f'HP dist plot: all HP values for {cat} are equal, '
                                      'no dist plot possible. Ignoring')
                else:
                    dist_hps.append(cat)
            if dist_hps:
                dist_df = df[df[self.CATEGORY_COL].isin(dist_hps)]
                #plot the HP distributions
                g = sns.FacetGrid(dist_df, row=self.CATEGORY_COL, height=1.7, aspect=4, 
                                  sharex=False, sharey=False)
                g.map(sns.distplot, self.VALUE_COL, color=color, hist=True, rug=True)
                g.set_titles(row_template="{row_name}")
                g.set(xlabel=None)
                f_name = 'HP_distributions'
                g.fig.canvas.set_window_title(f_name)
                
                if self.save_plots:
                    out_path = Path(self.output_dir) / f'{f_name}.png'
                    g.savefig(str(out_path)) #FacetGrid has savefig()
                
                handles.append(g)
        
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

        return handles


def input_path2analysis(pathOrAnalysisList):
    """NOTE: the given list is assumed to be in correct task order, IFF
    Analysis objects or multiple paths to experiment files are given!
    The task id is retireved from directory names only if ONE base dir is given
    that contains multiple experiment state files.
    """
    if not isinstance(pathOrAnalysisList, (list,tuple)):
        pathOrAnalysisList = [pathOrAnalysisList]
    if all([isinstance(e, tune.Analysis) for e in pathOrAnalysisList]):
        pass
    elif all([isinstance(e, (str, Path)) for e in pathOrAnalysisList]): 
        expObj_list = []
        for e in pathOrAnalysisList:
            dirOrExpFile = Path(e)
            assert dirOrExpFile.exists(), f'Element in `pathOrAnalysisList` was a string but not a valid path: {e}'
            #note this needs you to manually change the logdir in the EXPERIMENT_STATE_FILE Json's 
            # if the experiment is copied to somewhere
            if dirOrExpFile.is_file():
                expObj_list.append(tune.ExperimentAnalysis(str(dirOrExpFile)))
            else:
                # expObj_list.extend([tune.ExperimentAnalysis(str(p)) for p in dirOrExpFile.rglob(EXPERIMENT_STATE_FILE)])
                exp_files = list(dirOrExpFile.rglob(EXPERIMENT_STATE_FILE))
                if len(exp_files) == 1:
                    expObj_list.append(tune.ExperimentAnalysis(str(exp_files[0])))
                else:
                    sorting_list = []
                    for p in exp_files:
                        analysis = tune.ExperimentAnalysis(str(p))
                        try:
                            task_id = int(re.search(r'^T(\d+)_', Path(analysis._experiment_dir).name).group(1))
                        except (ValueError, IndexError, AttributeError) as e:
                            raise RuntimeError(f'Can not determine the task ID for {analysis._experiment_dir}! The ID is required '
                                                f'for sorting the tasks. \nTraceback: {e}')
                        if sorting_list and task_id in [e[0] for e in sorting_list]:
                            raise RuntimeError(f'The task_id was determined to {task_id} but a task already exists for that ID! '
                                f'Check the task dirs under {dirOrExpFile} and make sure no two dirs start with the same task number.')
                        sorting_list.append( (task_id, analysis) )
                    expObj_list.extend([a[1] for a in sorted(sorting_list, key=lambda x: x[0])])

        pathOrAnalysisList = expObj_list
    else:
        raise ValueError('Given input `pathOrAnalysisList` is neither a list of tune.Analysis objects nor strings!')

    return pathOrAnalysisList



def run_trial_plotter(input_path, output=None, override=False, logger=None):
    """Creates a default overview of plots from the Tune experiment
    output of one or multiple individual Trials (progress CSV files).
    Each Trial will get its own plots, regardless if they belong to the
    same Tune experiment.
    """

    logger = logger or setup_logger(__name__, verbose=True)

    #construct the input and output paths
    #FIXME: this should be handled by a Plotter class
    path = Path(input_path).absolute()
    if not path.is_file() or path.suffix != '.csv':
        logger.info('Input is not a CSV file. Scanning for progress.csv files in `input_path` as base dir...')
        expResult_files = list(path.rglob('progress.csv'))
        if not expResult_files:
            raise ValueError('No experiment result CSV file found to process!')

        output_dirs = []
        for p in expResult_files:
            # date_match = re.search(r'(\d+(-|\/)\d+(-|\/)\d+)', p.parent.name).group(0)
            # dir_name = 'plots_'+ p.parent.name.split(date_match if date_match else '_')[:-1][:-1]
            dir_name = 'plots_'+p.parent.name       #use the full dir name for now to distinguish betw HP sweeps in same env
            if not output:
                p_out = p.parent.parent / dir_name
            else:
                p_out = Path(output).absolute() / dir_name
            
            p_out = make_new_dir(p_out, override, logger)
            output_dirs.append(p_out)
        
    else:
        expResult_files = [path]
        output_dirs = [Path(output).absolute()] if output else [path.parent.parent / 'plots_'+path.parent.name]

    logger.info('Found {} CSV files to process:\n{}'.format(len(expResult_files),
        '\n\t'.join([f'{i} --> {o}' for i,o in zip(expResult_files,output_dirs)])))

    for csv_file,output_path in zip(expResult_files, output_dirs):
        logger.info('')
        logger.info(''.join(['-']*30))
        logger.info(f'Processing experiment file: {csv_file}')
        logger.info(f'Output will be stored at: {output_path}')

        plotter = TuneExperimentPlotter(str(output_path), str(csv_file), plot_style='darkgrid', logger=logger)
        # plotter.save_melted_df(str(output_path / 'melted_progress.csv'))

        curr_output = str(output_path / 'hypers')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output): 
            options = {'plot_dist':True, 
                    'color':'xkcd:dark magenta', 
                    'height':3, 
                    'aspect':1.5}
            try:
                plotter.plot_hypers(['Training Iterations','Wall Time [s]'], None, **options)
            except Exception as e:
                logger.warning(f'Failed to plot: Hyperparameters!\nTraceback: {e}')

        curr_output = str(output_path / 'hists')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output):
            options = {'aspect':2}
            try:
                plotter.plot_hists(**options)
            except Exception as e:
                logger.warning(f'Failed to plot: Hist data!\nTraceback: {e}')

        #should plot the same as plot_hists()
        curr_output = str(output_path / 'rewards')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output): 
            # with sns.plotting_context("poster"): #, font_scale=1.5
            options = {'color':'xkcd:teal', 
                    'height':4, 
                    'aspect':3}
            try:
                h = plotter.plot_data(['Timesteps','Training Iterations','Wall Time [s]'], 
                                    'hist_stats/episode_reward', 
                                    plot_type='single', 
                                    plot_style="line_sd",
                                    **options)
            except Exception as e:
                logger.warning(f'Failed to plot: Rewards!\nTraceback: {e}')
        
        curr_output = str(output_path / 'ep_stats')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output): 
            options = {'color':'xkcd:burnt orange', 
                    'linestyle': '-.',
                    # 'marker': 'x',     #when using plot_style="scatter"
                    # 'edgecolors':'none',    #when using plot_style="scatter"
                    # 's': 15,      #when using plot_style="scatter"
                    'height':3, 
                    'aspect':2}
            try:
                h = plotter.plot_data('Training Iterations', 'Episode Stats', 
                                    plot_type='single', 
                                    plot_style="line", 
                                    **options)
            except Exception as e:
                logger.warning(f'Failed to plot: Episode stats!\nTraceback: {e}')
        
        curr_output = str(output_path / 'hw_util')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output): 
            options = {'palette': 'Set1',
                    'style': plotter.CATEGORY_COL,
                    'aspect': 3,
                    'height': 4}
            try:
                h = plotter.plot_data('time_total_s', 
                                    ['CPU Utilization [perc]',
                                    'GPU Utilization [perc]',
                                    'RAM Utilization [perc]'], 
                                    plot_type='multi', 
                                    plot_style="line", 
                                    **options)
            except Exception as e:
                logger.warning(f'Failed to plot: HW utilization!\nTraceback: {e}')

        curr_output = str(output_path / 'losses')
        with plotter.context(show_plots=False, save_plots=True, output_dir=curr_output): 
            options = {'color':'limegreen', 
                    'height':4, 
                    'aspect':1}
            try:
                h = plotter.plot_data(['Training Iterations','Timesteps'], 'Losses', 
                                    plot_type='adjacent', 
                                    plot_style="line", 
                                    **options)
            except Exception as e:
                logger.warning(f'Failed to plot: Losses!\nTraceback: {e}')
    
    logger.info(''.join(['-']*30))
    logger.info(f'Plot creation   <<DONE>>   created plots from {len(expResult_files)} experiments')



def run_exp_average_plotter(analysis_list, output=None, override=False, logger=None, **plotArgs):
    """Creates a default overview of plots from a list of Tune Analysis
    objects and summarizes the mean and std of all Trials in a single experiment.
    Unless it is intended to average the experiment results over different HP settings,
    this should only be called on experiment dirs with same HP configuration (for example
    to see mean performance of agents with same setup for statistical significance and 
    reduced variance).
    """
    #header structure to extract and rename from the Analysis DF from each experiment
    headerStruct = {
        'Index':{
            'Timesteps': 'timesteps_total',
            # 'Episodes': 'episodes_total',
            'Training Iterations': 'training_iteration',
            'Wall Time [s]': 'time_total_s'},
        'Rewards':{
            'Mean Reward': 'episode_reward_mean',
            'Max Reward': 'episode_reward_max',
            'Min Reward': 'episode_reward_min'},
        # 'Hyperparams':{
        #     'Learning Rate': 'info/learner/default_policy/cur_lr',
        #     'Explained Variance': 'info/learner/default_policy/vf_explained_var',
        #     'KL Divergence': 'info/learner/default_policy/kl',
        #     'KL Coefficient': 'info/learner/default_policy/cur_kl_coeff',
        #     'Entropy': 'info/learner/default_policy/entropy',
        #     'Entropy Coefficient': 'info/learner/default_policy/entropy_coeff'},
    }
    
    logger = logger or setup_logger(__name__, verbose=True)

    save_plots = bool(plotArgs.pop('save', True))      #default True
    show_plots = bool(plotArgs.pop('show', False))     #default False

    # preproc input data to correct type, can handle Analysis obj's or paths
    analysis_list = input_path2analysis(analysis_list)

    logger.info(f'Processing {len(analysis_list)} experiments...')
    for analysis in analysis_list:
        if not isinstance(analysis, Analysis):
            raise ValueError(f'Experiment plotter needs a list of tune.Analysis-like objects, got a {type(analysis)}')
        
        nbOfTrials = len(analysis.trial_dataframes)
        exp_basedir = Path(analysis._experiment_dir)
        exp_name = exp_basedir.name
        logger.info(''.join(['-']*30))
        logger.info(f'Processing experiment \'{exp_name}\' with {nbOfTrials} individual Trials.')
        logger.debug(f'The Experiment average at each point will be calculated from {nbOfTrials} values.')
        if nbOfTrials < 1:
            logger.error('No Trials for the current experiment found! Can not plot any values!\n\n')
            continue
        elif nbOfTrials == 1:
            logger.warning('Analysis has only one Trial for the experiment! Averaging '
                           'statistics like mean / std are disabled!')
            ci = None
        else:
            ci = "sd"
        
        outdir_name = f'plots_Exp-{exp_name}'
        if output is None:
            out_path = exp_basedir / outdir_name
        else:
            out_path = Path(output) / outdir_name
        
        #create or clear the output dir
        out_path = make_new_dir(out_path, override, logger)
        logger.info(f'Input/Output mapping: {exp_basedir}  -->  {out_path}')

        # concat all the trial DF
        name_mappings = {v:k for cat,maps in headerStruct.items() for k,v in maps.items()}
        select_cols = list(name_mappings.keys())
        df = pd.concat([trial_df for trial_df in analysis.trial_dataframes.values()], ignore_index=True)
        df = df[select_cols]    #remove all columns not in headerStruct 

        # rename the id column names according to headerStruct
        df.rename(columns=name_mappings, errors="raise", inplace=True)
        #remove invalid data and convert to float type
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # df = df.astype(np.float)

        # start plotting
        sns.set(style='darkgrid')
        try:
            # only use categories that were given
            xNames = [x for cat,maps in headerStruct.items() for x in maps.keys() if cat == 'Index']
            yNames = [y for cat,maps in headerStruct.items() for y in maps.keys() if cat != 'Index']
            logger.info('Creating {} figures...'.format(len(xNames)*len(yNames)))
            for xName in xNames:
                for yName in yNames:
                    # if trial_selection == 'best':
                    #     g = sns.relplot(xName, yName, hue='Task', kind="line", ci=ci,
                    #                     # palette=["b", "r"],
                    #                     data=df, **plotArgs)
                    # elif trial_selection == 'all':
                    #     g = sns.lineplot(x=xName, y=yName, hue='Task', units="Trial",
                    #                     estimator=None, lw=1,
                    #                     # palette=["b", "r"],
                    #                     data=experiment_df, **plotArgs)
                    g = sns.relplot(x=xName, y=yName, kind="line", ci=ci, data=df, **plotArgs)
                    g.set(ylabel=f'{yName}\n({nbOfTrials} Trials)')
                    xName_p = xName.replace(' ','').replace(']','').replace('[','_')
                    yName_p = yName.replace(' ','').replace(']','').replace('[','_')
                    f_name = f'{exp_name}_{xName_p}_{yName_p}'
                    g.fig.canvas.set_window_title(f_name)
                    
                    if save_plots:
                        file_path = Path(out_path) / f'{f_name}.png'
                        g.savefig(str(file_path)) #relplot returns a FacetGrid
                    
        except Exception as e:
            logger.error(f'Error while creating averaging plots from experiment {exp_name}. '
                         f'Occurred during plot for {yName} over {xName}.\nTraceback: {e}')
            continue
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')


def run_task_progress_plotter(analysis_list, task_perf_dict=None,
                              trial_selection='best',
                              output=None, override=False, 
                              save_csv=False, logger=None, 
                              **plotArgs):
    """Creates a default overview of plots from a list of Tune Analysis
    objects and summarizes the mean and std of all Trials in a single experiment.
    Unless it is intended to average the experiment results over different HP settings,
    this should only be called on experiment dirs with same HP configuration (for example
    to see mean performance of agents with same setup for statistical significance and 
    reduced variance).
    If task_perf_dict is given, it needs to be a dict with 
        {task ID: [min , max achievable performance values]}
    Then the values for each task will be normalized to make the graph comparable.
    """
    #header structure to extract and rename from the Analysis DF from each experiment
    headerStruct = {
        'Index':{
            'Timesteps': 'timesteps_total',
            # 'Episodes': 'episodes_total',
            'Training Iterations': 'training_iteration',
            'Wall Time [s]': 'time_total_s'
        },
        'Rewards':{
            'Mean Reward': 'episode_reward_mean',
            'Max Reward': 'episode_reward_max',
            'Min Reward': 'episode_reward_min'
        },
        'Hist Data':{
            'Reward / Episode': 'hist_stats/episode_reward',
            # 'Episode Lengths': 'hist_stats/episode_lengths'
        },
    }

    logger = logger or setup_logger(__name__, verbose=True)
    logger.info('Running multi-task plotting...')

    # preproc input data to correct type, can handle Analysis obj's or paths
    analysis_list = input_path2analysis(analysis_list)
   
    task_basedir = Path(analysis_list[-1]._experiment_dir)
    if len(analysis_list) > 1:
        task_basedir = task_basedir.parent
        #check that all analysis objs belong to the same multi-task experiment
        # if not all([str(Path(e._experiment_dir).parent) == str(task_basedir) for e in analysis_list]):
        #     raise ValueError('Not all experiments/tasks belong to the same run! (Do not share the same basedir)')
    train_name = task_basedir.name
    outdir_name = f'plots_{train_name}'
    if not output:
        out_path = task_basedir / outdir_name
    else:
        out_path = Path(output) / outdir_name
    
    #create or clear the output dir
    out_path = make_new_dir(out_path, override, logger)
    logger.info(f'Input/Output mapping: {task_basedir}/*  -->  {out_path}')

    #pop some of the plot options not belonging to seaborn/matplotlib
    save_plots = bool(plotArgs.pop('save', True))      #default True
    show_plots = bool(plotArgs.pop('show', False))     #default False

    nbOfTasks = len(analysis_list)
    logger.info(f'Processing {nbOfTasks} tasks (experiments)...')
    # 1. get the DF's from the analysis obj, determine task order and preprocess the DF values
    logger.info('Retrieving metrics for each experiment based on Trial selection...')
    max_trial_len = 0
    sorted_taskDf = []
    for task_id, analysis in enumerate(analysis_list):
        nbOfTrials = len(analysis.trial_dataframes)
        if nbOfTrials < 1:
            raise ValueError('No Trials for the current experiment found! Can not plot any values!')

        name_mappings = {v:k for cat,maps in headerStruct.items() for k,v in maps.items()
                        if cat in ['Index']}
        
        metric = 'episode_reward_mean'
        best_trial_dir = tune.Analysis.get_best_logdir(analysis, metric=metric, mode='max')
        if trial_selection == 'best':
            logger.info(f'Pick strategy for Task {task_id}: Best trial of the task ({os.path.basename(best_trial_dir)})')
            from ast import literal_eval
            df = analysis.trial_dataframes[best_trial_dir].copy()
            max_trial_len = max(max_trial_len, len(df))
            #since we take only one DF for each task, we need to use the hist stats for the reward
            # metric to be able to calc mean and std
            y_cols = list(headerStruct['Hist Data'].values())
            # convert the list strings in the df into actual list objects
            for ycol in y_cols:
                # df[ycol].update(df[ycol].apply(literal_eval))
                df = df.assign(**{ycol:df[ycol].apply(literal_eval)})
                # df = df.explode(ycol, ignore_index=True)   #expand the list entries to own rows, i.e. create the long dataframe
            #reduce data usage of frame before exploding
            dtype_mapping = {'float64': 'float16', 'int64':'int32'}
            for d_from,d_to in dtype_mapping.items():
                sel_cols = df.select_dtypes(include=d_from)
                df.loc[:,sel_cols.columns] = sel_cols.astype(d_to)
            df = df.apply(lambda x: x.explode() if x.name in y_cols else x)
            df = df.dropna(subset=y_cols)
            for ycol in y_cols:
                df.loc[:,ycol] = df[ycol].infer_objects()

            # df = df.astype({c:np.dtype('float64') for c in y_cols})

            df['Trial'] = f'best_{metric}'
            df['best_trial'] = 'yes'
            
        elif trial_selection == 'all':
            logger.info(f'Pick strategy for Task {task_id}: Showing all trials of the task')
            y_cols = list(headerStruct['Rewards'].values())
            df = pd.DataFrame()
            for trial_id, (trial_dir,trial_df) in enumerate(analysis.trial_dataframes.items()):
                tmp = trial_df.copy()
                tmp['Trial'] = f'Trial {trial_id}'
                if trial_dir == best_trial_dir:
                    tmp['best_trial'] = 'yes'
                else:
                    tmp['best_trial'] = 'no'
                max_trial_len = max(max_trial_len, len(tmp))
                df = df.append(tmp, ignore_index=True)
            # df = pd.concat([trial_df for trial_df in analysis.trial_dataframes.values()], ignore_index=True)
        
        else: 
            raise ValueError(f'Unkown trial selection scheme: {trial_selection}')
        
        name_mappings.update(
            {v: f'{k} Performance' if task_perf_dict else k 
                for cat,maps in headerStruct.items() 
                    for k,v in maps.items() if v in y_cols}
        )

        if task_perf_dict:
            #normalize the y col(s)
            logger.debug('Normalizing y-axis to performance [%] score')
            perf_vals = task_perf_dict[task_id]
            for ycol in y_cols:
                df[ycol] = df[ycol].sub(perf_vals[0])
                df[ycol] = df[ycol].div(perf_vals[1]-perf_vals[0])
                if df[ycol].max() > 1:
                    logger.warning(f'The max. performance value given for task {task_id}, Col. {ycol} is too low. Performance is sometimes over 100%!')
        
        # keep only the needed cols
        select_cols = list(name_mappings.keys()) + ['Task', 'Trial', 'best_trial']
        df['Task'] = f'Task {task_id}'
        df = df[select_cols].copy()    #remove all columns not in headerStruct

        # rename the id column names according to headerStruct
        df.rename(columns=name_mappings, errors="raise", inplace=True)
        #remove invalid data and convert to float type
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        sorted_taskDf.append((task_id, df))
    
    sorted_taskDf = [e[1] for e in sorted(sorted_taskDf, key=lambda x: x[0])]

    # only use categories that were given
    xNames = [x for orig,x in name_mappings.items() if orig in list(headerStruct['Index'].values())]
    yNames = [y for orig,y in name_mappings.items() if orig not in list(headerStruct['Index'].values())]
    # 2. now with correct order, concatenate the task-DF's and restore the 'Index' values
    logger.info('Concentrating task metrics into single dataframe and determining task boundaries...')
    task_boundary_indexes = []  #x values UP TO which the current task was trained
    x_offset = {name:0 for name in xNames}
    experiment_df = pd.DataFrame()
    for task_df in sorted_taskDf:
        for xName in xNames:
            task_df[xName] = task_df[xName].add(x_offset[xName])
            x_offset[xName] = task_df[xName].max()
        #these determine the positions of the vlines for the tasks
        task_boundary_indexes.append(copy.deepcopy(x_offset))
        experiment_df = experiment_df.append(task_df, ignore_index=True)

    if save_csv:
        csv_file = str(out_path / 'multi-task.csv')
        logger.info(f'Saving resulting DataFrame for the whole experiment to: {csv_file}')
        experiment_df.to_csv(csv_file, index=False, na_rep='nan')
    
    #apply filter to long dataframes
    if trial_selection == 'all':
        def cut_edge_from(mask, cut_len, back=False):
            c = 0
            if back: mask = mask[::-1]
            for i,f in enumerate(mask):
                if f:
                    c+=1
                    mask.iloc[i] = ~mask.iloc[i]
                if c >= cut_len:
                    break
            if back: mask = mask[::-1]
            return mask
        
        filter_mode = 'median'
        logger.info(f'Applying data {filter_mode} filter to trial dataframes...')
        if max_trial_len > 10000:   #condition when to smooth the data: more than 10k datapoints
            for task in experiment_df['Task'].unique():
                for trial in experiment_df[experiment_df['Task']==task]['Trial'].unique():
                    #get the sub-df for each task and trial
                    # tt_df = experiment_df[experiment_df['Task']==task & experiment_df['Trial']==trial]
                    filter_length = int(max_trial_len * 0.01)
                    for yName in yNames:
                        mask = (experiment_df['Task']==task) & (experiment_df['Trial']==trial)
                        if filter_mode == 'mean':
                            filter_cutoff = int(filter_length*0.3)
                            filtr_data =\
                                np.convolve(experiment_df.loc[mask, yName], np.ones((filter_length)), mode='same') / filter_length
                        elif filter_mode == 'median':
                            if filter_length % 2 == 0: filter_length -= 1   #filter len must be odd
                            filter_cutoff = int(filter_length*0.1)
                            filtr_data = medfilt(experiment_df.loc[mask, yName], filter_length)
                        #cut the edges
                        if filter_cutoff:
                            mask = cut_edge_from(mask, filter_cutoff, False)
                            mask = cut_edge_from(mask, filter_cutoff, True)
                            filtr_data = filtr_data[filter_cutoff:-filter_cutoff]

                        experiment_df.loc[mask, yName] = filtr_data

    # 3. plot the concentrated DF and task boundaries
    logger.info('Creating {} multi-task progress plots...'.format(len(xNames)*len(yNames)))
    sns.set(style='darkgrid')
    if len(analysis_list)<=1:
        palette = [plotArgs.pop('color')] if plotArgs.get('color') else None
    else:
        palette = plotArgs.pop('palette', None)
    try:
        for xName in xNames:
            for yName in yNames:
                if trial_selection == 'best':
                    g = sns.relplot(x=xName, y=yName, hue='Task', kind="line", ci='sd',
                                    palette=palette,
                                    data=experiment_df, **plotArgs)
                elif trial_selection == 'all':
                    g = sns.relplot(x=xName, y=yName, hue='Task', units="Trial",
                                    estimator=None, #lw=1,
                                    kind="line", ci=None,
                                    size='best_trial', sizes={'yes': 1.8, 'no': 0.7},
                                    palette=palette,
                                    data=experiment_df, **plotArgs)
                
                ax = g.axes[0,0]    #equivalent to plt.gca() if only one subplot
                if len(task_boundary_indexes) > 1:
                    #add the task boundary v-lines if more than one task is plotted
                    ty = ax.get_ylim()[1] * 0.90
                    for tid,bound in enumerate(task_boundary_indexes):
                        xpos = bound[xName]
                        tx = task_boundary_indexes[tid-1][xName] if tid > 0 else 0. #text right of task boundary
                        tx += 0.015 * float(experiment_df[xName].max())
                        plt.axvline(xpos, 0, 1, color='dimgrey',linewidth=1, linestyle='--')
                        plt.text(tx, ty, f'Task {tid}', fontsize='medium', 
                                color='darkslategray', ha='left', va='center',)   #darkslategray dimgray
                                #  transform=ax.transAxes)
                
                g.set(ylabel=f'{yName}')
                xName_p = xName.replace(' ','').replace(']','').replace('[','_').replace('/','per')
                yName_p = yName.replace(' ','').replace(']','').replace('[','_').replace('/','per')
                f_name = f'{train_name}_{xName_p}_{yName_p}'
                g.fig.canvas.set_window_title(f_name)
                
                if save_plots:
                    file_path = Path(out_path) / f'{f_name}.png'
                    g.savefig(str(file_path)) #relplot returns a FacetGrid
                
    except Exception as e:
        logger.error(f'Error while creating multi-task plot from experiment {train_name}. '
                        f'Occurred during plot for {yName} over {xName}.\nTraceback: {e}')
        raise Exception from e

    if show_plots:
        plt.show()
    plt.close('all')



# test_file = '/home/dennis/tune_out/st_Pongv4_PNN/st_Pongv4_PNN_2020-07-14/PPO_PongNoFrameskip-v4_0_2020-07-14_03-34-36fscuzflj/progress.csv'
test_file = '/home/dennis/tune_out/multi_task/PNN_pong_soup_2/T2_Pong_flipHorizontal/PPO_Pong_flipHorizontal_0_2020-07-31_23-19-07vdft07r0/progress.csv'
if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', dest='input', type=str, nargs='*', default=[])
    parser.add_argument('-o', '--output', dest='output', type=str, default='')
    parser.add_argument('-fn', '--function', dest='function', type=str, default='trial_plot', 
                        choices=['trial_plot', 'exp_plot', 'task_plot'])
    parser.add_argument('--task_min_max', dest='tmm', type=yaml.load, default="{}")
    parser.add_argument('--trial_selection', dest='trial_selection', type=str, default="best",
                        choices=['best', 'all'])
    parser.add_argument('--override', dest='override', action="store_true", default=False)
    parser.add_argument('--debug', dest='_debug', action="store_true", default=False)
    args = parser.parse_args()

    if not args._debug:

        if args.function == 'trial_plot':
            for in_path in args.input:
                p = Path(in_path)
                if not p.is_dir():
                    print(f'\n\nValue Error: Given input path {p} is not a directory. '
                            'A directory is needed as input path for trial_plot. Skipping...')
                    continue
                run_trial_plotter(p, args.output or None, args.override, None)
        
        elif args.function == 'exp_plot':
            #note this needs you to manually change the logdir in the experiment_state*.json's 
            # if the experiment was copied to somewhere
            #NOTE: new fn for exp plotting
            # this works a bit different than the old funktion: 
            #       if trial_selection=all: show all experiments and mark best one thicker
            #       else: select BEST trial from experiment and plot mean-std of hist data
            exp_file_list = []
            for in_path in args.input:
                p = Path(in_path)
                if p.is_file():
                    exp_file_list.append(p.parent)
                else:
                    exp_file_list.append(p)
            exp_file_list = list(Path(args.input).rglob(EXPERIMENT_STATE_FILE))
            if not exp_file_list:
                raise ValueError(f'Could not find any experiment state file ({EXPERIMENT_STATE_FILE}) at: {args.input}')
            for exp_state_info in exp_file_list:
                exp_dir = exp_state_info.parent
                options = {'height':4, 
                        'aspect':3,
                        'color':'blue',
                        'legend':'full', #'brief', 'full', or False
                }
                run_task_progress_plotter(exp_dir, args.tmm, args.trial_selection,
                                        args.output or None, 
                                        args.override, False, 
                                        None, **options)
            #NOTE: old fn for exp plotting
            # plots the mean-std of each metric from all trials
            # options = {'color':'xkcd:blood red', 
            #         'height':3, 
            #         'aspect':2}
            # run_exp_average_plotter(args.input, args.output or None, args.override, None, **options)
        
        elif args.function == 'task_plot':
            #note this needs you to manually change the logdir in the experiment_state*.json's 
            # if the experiment was copied to somewhere
            options = {'height':4, 'aspect':3, 'palette':None}
            run_task_progress_plotter(args.input, args.tmm, args.trial_selection,
                                      args.output or None, 
                                      args.override, False, 
                                      None, **options)

    else:
        #run whatever you want to debug here
        path = '/home/dennis/tune_out/PNN_Pong-Alien_lr/'
        # output = '/home/dennis/tune_out/plot_test'
        perf_dict = {
            0: [-21., 21.],
            1: [0., 3000.]  #max reward for Alien goes up to 4300, but mean is around ~1000!
        }
        perf_dict = None
        options = {'color':'r', 
                   'height':4, 
                   'aspect':3,
                   'legend':False}
        run_task_progress_plotter(path, perf_dict, 'all', args.output or None, 
                                override=False, save_csv=False, logger=None, **options)
    