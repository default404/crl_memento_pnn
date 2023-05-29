# small sub-module to test the PNN implementation based on supervised learning
# on permuted MNIST
import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
__PRJ_DIR__ = os.path.dirname(os.path.dirname(__MAINDIR__))
if not __PRJ_DIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __PRJ_DIR__)

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import logging
import visdom

from torch.autograd import Variable
from tqdm import tqdm


from tests.test_pnn_supervised.src.data.PermutedMNIST import get_permuted_MNIST
# from src.model.ProgressiveNeuralNetworks import PNN
from tests.test_pnn_supervised.src.tools.arg_parser_actions import LengthCheckAction
from tests.test_pnn_supervised.src.tools.evaluation import evaluate_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import gym
from common.util import load_config_and_update, convert_to_tune_config
from continual_atari.agents.policies.pnn.pnn_model import PNN

__PRJ_DIR__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))))
_DEFAULT_CONFIG_PATH = Path(__PRJ_DIR__) / "continual_atari/configs/default_config.yml"

def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-path', default='/home/dennis/Git_Repos/pnntest', type=str, help='path to the data')
    parser.add_argument('-cuda', default=-1, type=int, help='Cuda device to use (-1 for none)')
    parser.add_argument('-visdom_url', default="localhost", type=str, help='Visdom server url')
    parser.add_argument('-visdom_port', default=8097, type=int, help='Visdom server port')

    parser.add_argument('--layers', metavar='L', type=int, default=3, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[784, 1024, 512], nargs='+',
                        action=LengthCheckAction)

    parser.add_argument('--n_tasks', dest='n_tasks', type=int, default=2)
    parser.add_argument('--epochs', dest='epochs', type=int, default=1)
    parser.add_argument('--bs', dest='batch_size', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4, help='Optimizer weight decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=1e-4, help='Optimizer momentum')

    args = parser.parse_known_args()
    return args[0]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])
    #start visdom server before running this with `python -m visdom.server` in the python env where visdom is installed
    viz = visdom.Visdom(server=args['visdom_url'], port=args['visdom_port'], env='PNN MNIST test')

    m_config = {
        "algorithm": "pnn",
        "Continual_params":{
            "algo_params":{
                "pnn":{
                    "conv_filters": [[16, [6, 6], 2], [32, [4, 4], 2], [64, [7, 7], 1]],
                    "fcnet_hiddens": [256]
                }
            }
        } 
    }
    default_config = load_config_and_update(_DEFAULT_CONFIG_PATH, updatesDict=m_config)
    config, _ = convert_to_tune_config(default_config)
    obs_space = gym.spaces.Box(np.zeros([28,28,1]), np.ones([28,28,1])*255, dtype=np.int8)
    action_space = gym.spaces.Discrete(10)
    num_outputs = int(10)
    model = PNN(obs_space, action_space, num_outputs, model_config=config['model'], name="PNN")

    tasks_data = [get_permuted_MNIST(args['path'], args['batch_size']) for _ in range(args['n_tasks'])]

    x = torch.Tensor()
    y = torch.LongTensor()

    if args['cuda'] != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args['cuda']))
        logger.info('Running with cuda (device: {})'.format(device))
    else:
        logger.warning('Running WITHOUT cuda')
        device = torch.device("cpu")
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    for task_id, (train_set, val_set, test_set) in enumerate(tasks_data):
        # val_perf = evaluate_model(model, x, y, val_set, task_id=task_id)

        model.freeze_columns()
        model.next_task(obs_space, action_space)

        optimizer = torch.optim.RMSprop(model.parameters(task_id), lr=args['lr'],
                                        weight_decay=args['wd'], momentum=args['momentum'])

        train_accs = []
        train_losses = []
        step = 0
        #initial visdom plot for live update
        win_acc = viz.line(X=np.array([None]), Y=np.array([None]),
                           opts={'title': 'Task {}: train/val accuracy'.format(task_id)})
        viz.line(X=np.array([step]),
                 Y=np.array([0]),  
                 win=win_acc, name='train acc', update='append',
                 opts={'showlegend':True})
        viz.line(X=np.array([step]),
                 Y=np.array([0]),  
                 win=win_acc, name='val acc', update='append',
                 opts={'showlegend':True})
        for epoch in range(args['epochs']):
            total_samples = 0.
            total_loss = 0.
            correct_samples = 0
            input_dict = {
                "obs": None,
                "prev_action": None,
                "prev_reward": 0,
                "is_training": True,
                "task_id": task_id
            }
            for inputs, labels in tqdm(train_set):
                step += 1
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                # x = x.view(x.size(0), -1)
                x = Variable(x)
                input_dict["obs"] = x
                predictions, _ = model(input_dict)
                input_dict["prev_action"] = predictions

                _, predicted = torch.max(predictions.data, 1)
                total_samples += y.size(0)
                correct_samples += (predicted == y).sum()

                indiv_loss = F.cross_entropy(predictions, Variable(y))
                total_loss += indiv_loss.item()

                optimizer.zero_grad()
                indiv_loss.backward()
                optimizer.step()

                #live update the train acc
                viz.line(X=np.array([step]),
                    Y=np.array([correct_samples / total_samples]),  
                    win=win_acc, name='train acc', update='append')

            train_accs.append(correct_samples / total_samples)
            train_losses.append(total_loss / total_samples)
            logger.info(
                '[T{}]:E[{}/{}] Loss={}, Acc= {}'.format(task_id, epoch, args['epochs'], train_losses[-1],
                                                       train_accs[-1]))
            viz.line(np.array(train_accs), X=np.arange(epoch+1), win='tacc{}'.format(task_id),
                     opts={'title': 'Task {}: epoch train accuracy'.format(task_id)})
            viz.line(np.array(train_losses), X=np.arange(epoch+1), win='tloss{}'.format(task_id),
                     opts={'title': 'Task {}: epoch train loss'.format(task_id)})
            #validation
            input_dict['is_training'] = False
            val_perf = evaluate_model(model, input_dict, x, y, val_set)
            viz.line(X=np.array([step]),
                     Y=np.array([val_perf]),  
                     win=win_acc, name='val acc', update='append')
            logger.info(
                '[T{}]:E[{}/{}] Val Acc= {}'.format(task_id, epoch, args['epochs'], val_perf))

        perfs = []
        logger.info('Evaluation after task {}:'.format(task_id))
        for i in range(task_id + 1):
            _, val, test = tasks_data[i]
            input_dict['is_training'] = False
            input_dict['task_id'] = i
            # val_perf = evaluate_model(model, input_dict, x, y, val, task_id=i)
            test_perf = evaluate_model(model, input_dict, x, y, test, task_id=i)
            perfs.append(test_perf)
            logger.info('\tT n°{} - test:{}'.format(i, test_perf))

        viz.line(np.array(perfs), X=np.arange(task_id+1), win='all_task',
                        opts={'title': 'Evaluation on all tasks', 'legend': ['Test']})

    #test checkpoint/model export / import
    f_name = 'weights_t{}.h5'.format(task_id)
    model.export_checkpoint('./saved', f_name)
    model.export_model('./saved')
    model.import_from_h5('./saved/{}'.format(f_name))


if __name__ == '__main__':
    main(vars(get_args()))
