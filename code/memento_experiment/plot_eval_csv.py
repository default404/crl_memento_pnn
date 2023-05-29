
import os, sys, inspect
__MEMENTO_DIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
__MAINDIR__ = os.path.dirname(__MEMENTO_DIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns

from common.util import setup_logger
from memento_experiment import EVAL_CSV_NAME, MEMENTO_CSV_NAME

# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__


def plot_eval_metrics(eval_df, plot_dir, yName='Episode Return', for_env=None):
    plot_dir = Path(plot_dir)
    sns.set(style='darkgrid')
    if yName not in eval_df.columns:
        raise ValueError(f'Given y-name {yName} not in the DF!')
    if for_env is not None:
        env_eval_df = eval_df.loc[eval_df['Environment'] == for_env]
    else:
        env_eval_df = eval_df
    env_eval_df.infer_objects()

    options = {'height':8, 
                'aspect':2,
                'legend':'brief', #'brief', 'full', or False
        }
    f_name = 'Timesteps_{}'.format(yName.replace(' ',''))
    # plot of all individual episode returns
    g = sns.relplot(x='Timesteps', y=yName, hue='Episode',
                    estimator=None, lw=0.5,
                    kind="line", ci=None,
                    data=env_eval_df, **options)
    file_path = plot_dir / f'all_{f_name}.png'
    g.savefig(str(file_path), dpi=300)
    # plot of a unified mean-std curve for all episodes
    # NOTE: can fail if some max rewards are only reached once! 
    options = {'height':4, 
                'aspect':3,
                'legend':False}
    g_mean = sns.relplot(x='Timesteps', y=yName, 
                    kind="line", ci='sd', data=env_eval_df, **options)
    file_path = plot_dir / f'mean_{f_name}.png'
    g_mean.savefig(str(file_path), dpi=300)


def plot_memento_metrics(memento_df, output_dir, for_env=None):
    output_dir = Path(output_dir)
    sns.set(style='darkgrid')
    if for_env is not None:
        #create plot on episodes for specific env only
        mem_df = (memento_df.loc[memento_df['Environment'] == for_env]).infer_objects()
        options = {'height':4, 'aspect':3, 'legend':False}
        # bar plot memento rewards over episodes
        ax_bar = sns.barplot(x='Episode', y='Return', ci=None,
                            color='tab:blue', data=mem_df)
        ax_bar.set_xticks([])
        ax_bar.set_xlabel('Episodes')
        file_path = output_dir / 'memento_return_bars.png'
        fig = ax_bar.get_figure()
        fig.savefig(str(file_path))
        # plot memento reward distribution over episodes
        g_dist = sns.displot(data=mem_df, x="Return", kde=True, **options)
        file_path = output_dir / 'memento_return_distribution.png'
        g_dist.savefig(str(file_path))

    else:
        #create merged plots for all envs 
        mem_df = memento_df.infer_objects()
        g_dist = sns.displot(data=mem_df, x="Return", hue='Task', kind='kde', height=4, aspect=3)
        file_path = output_dir / 'memento_return_dist_all_tasks.png'
        g_dist.savefig(str(file_path))
        g_dist = sns.displot(data=mem_df, x="Return", hue='Environment', kind='kde', height=4, aspect=3)
        file_path = output_dir / 'memento_return_dist_all_envs.png'
        g_dist.savefig(str(file_path))


def main(eval_folder, logger=None):
    if not logger:
        logger = setup_logger(name='Eval_Plotter', verbose=True)

    eval_folder = Path(eval_folder)
    if not eval_folder.is_dir():
        raise ValueError('Invalid eval_folder! Not a directory')

    eval_csv = list(eval_folder.glob(EVAL_CSV_NAME))
    if len(eval_csv) != 1 or not eval_csv[0].is_file():
        err_str = f'Could not find the evaluation CSV {EVAL_CSV_NAME}! Plots for this csv are disabled!'
        logger.warning(err_str)
        raise ValueError(err_str)
        eval_df = None
    else:
        eval_csv = eval_csv[0]
        eval_df = pd.read_csv(eval_csv)

    mem_csv = list(eval_folder.glob(MEMENTO_CSV_NAME))
    if len(mem_csv) != 1 or not mem_csv[0].is_file():
        err_str = f'Could not find the memento CSV {MEMENTO_CSV_NAME}! Plots for this csv are disabled!'
        logger.warning(err_str)
        mem_df = None
    else:
        mem_csv = mem_csv[0]
        mem_df = pd.read_csv(mem_csv)

    for env_id in eval_df['Environment'].unique():
        tid = int(eval_df.loc[eval_df['Environment'] == env_id, 'Task'][0])
        env_dir = eval_folder / f'{tid}_{env_id}'
        plot_dir = env_dir / 'plots'
        if not plot_dir.exists():
            os.makedirs(str(plot_dir))
        
        logger.info(f'Generating reward plots for the evaluated env {env_id}...')
        try:
            if mem_df is not None:
                plot_memento_metrics(mem_df, plot_dir, for_env=env_id)
            if eval_df is not None:
                plot_eval_metrics(eval_df, plot_dir, for_env=env_id)
        except Exception as e:
            logger.warning(f'During the plot of the evaluation for env {env_id} an error occoured!\n'
                            f'Traceback: {type(e)}: {e}')
    if mem_df is not None:
        logger.info('Plotting summary distributions of each env of the state buffer...')
        try:
            plot_memento_metrics(mem_df, eval_folder, for_env=None)
        except Exception as e:
            logger.warning('During the summary distribution plots an error occoured!\n'
                            f'Traceback: {e}')


# input args
parser = argparse.ArgumentParser()
parser.add_argument('--eval_folder', dest='eval_folder', type=str, required=True)


if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        args = parser.parse_args()
        main(args.eval_folder)
    finally:
        os.chdir(__ORIG_CWD__)