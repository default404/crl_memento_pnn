import os, errno, logging, pickle
import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import same_padding, normc_initializer, SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

# The name of the env spec pickle file that is exported together with the
# model checkpoint when the PNN is stored.
# Needed for restoring a PNN with multiple pre-trained columns.
ENV_SPEC_SAVE_FILE = 'pnn_env.spec'


#NOTE: this should actually got to some kind of utility module
from ray.rllib.models import ModelCatalog
def get_action_dist(action_space,
                    config,
                    dist_type=None,
                    framework="torch",
                    **kwargs):
    
    dist, action_dim = ModelCatalog.get_action_dist(action_space,
                                                    config,
                                                    dist_type=dist_type,
                                                    framework=framework,
                                                    kwargs=kwargs)
    return dist, action_dim


#TODO: incorporate batchnorm (and maybe dropout), BUT Rllib seems not to differentiate btw
# train / eval mode (model.train()/.eval()) for Torch, sampling is only done with decorator 
# torch.no_grad(). Then these layers will work wrong
class PNNLinearBlock(nn.Module):
    '''A fully conntected linear layer that also assigns a set of 
    `adapter networks` to each of its instances depending on the current column.  
    Objects keep track of vertically assigned inputs from previous colums in a ModuleList.
    '''
    def __init__(self, 
                 col, 
                 depth, 
                 n_in, 
                 n_out, 
                 activation=None,
                 initializer=None,
                 alpha=None):
        super(PNNLinearBlock, self).__init__()
        self.col = col
        self.depth = depth
        self.n_in = n_in
        self.n_out = n_out
        alpha = alpha if alpha is not None else np.random.choice([1., .1, .01])
        self.W = SlimFC(n_in, n_out,
                        activation_fn=None,
                        initializer=initializer)
        if callable(activation) and isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

        self.V = nn.ModuleList()        #projection matrix of interior MLP
        self.alphas = nn.ParameterList()    #learnable scaling parameter of interior MLP
        if self.depth > 0:  #if the block is not in the first layer
            self.U = SlimFC(n_in, n_out,
                            activation_fn=None,
                            initializer=initializer) #projection matrix from prev layer as from Eq. 1
            # actually input[:-1] should be concatinated and mult. by a single big matrix V;
            # since it can be shown that `V_1*x_1+V_2*x_2 = V_1:2*x_1:2` it's sufficient to 
            # apply `V * alpha * x` column-wise and sum.
            # NOTE: this requires the same architecture for all columns since all prev column
            # outputs must have size `n_in`
            self.V.extend([SlimFC(n_in, n_in,
                                  activation_fn=None,
                                  initializer=initializer) 
                           for _ in range(col)])
            #NOTE: its not clear from paper if alpha is parameter vector with `n_in` learnable
            # values or a single scalar for complete MLP adapter
            # self.alphas.extend([nn.Parameter(torch.tensor(np.random.choice([1, .1, .01])), requires_grad=True) 
            #                    for _ in range(col)])
            self.alphas.extend([nn.Parameter(torch.tensor(alpha), requires_grad=True) 
                               for _ in range(col)])


    def forward(self, inputs):
        """this forward solution uses adapters with non-linear mapping and learned scalars
        as proposed by orig paper, eq. 2; instead of linear mapping of eq. 1.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.W(inputs[-1])
        prev_columns_out = [v( a * x ) for v,a,x in zip(self.V, self.alphas, inputs)]
        if prev_columns_out:
            return self.activation(cur_column_out + self.U(self.activation(sum(prev_columns_out))))
        else:
            #adapters_out = torch.zeros(1)
            return self.activation(cur_column_out)
        
        #return self.activation(cur_column_out + adapters_out)

    # def forward(self, inputs):
    #     """this forward solution uses the original lateral connections as 
    #     proposed by orig paper, eq. 1 (no non-linear or learned scalars).
    #     """
    #     if not isinstance(inputs, list):
    #         inputs = [inputs]
    #     cur_column_out = self.W(inputs[-1])
    #     prev_columns_out = [mod(x) for mod, x in zip(self.V, inputs)]
    #     return self.activation(cur_column_out + self.U(sum(prev_columns_out)))


class PNNConvBlock(nn.Module):
    '''A fully conntected linear layer that also assigns a set of 
    `adapter networks` to each of its instances depending on the current column.  
    Objects keep track of vertically assigned inputs from previous colums in a ModuleList.
    '''
    def __init__(self, 
                 col, 
                 depth, 
                 ch_in, 
                 ch_out, 
                 kernel, 
                 stride, 
                 padding, 
                 activation=None,
                 initializer="default",
                 alpha=None):
        super(PNNConvBlock, self).__init__()
        self.col = col
        self.depth = depth
        self.ch_in = ch_in
        self.ch_out = ch_out
        alpha = alpha if alpha is not None else np.random.choice([1., .1, .01])
        
        if callable(activation) and isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

        self.W = SlimConv2d(ch_in, ch_out, 
                            kernel, stride, padding, 
                            activation_fn=None,
                            initializer=initializer)

        self.V = nn.ModuleList()        #projection matrix of interior MLP
        self.alphas = nn.ParameterList()    #learnable scaling parameter of interior MLP
        if self.depth > 0:  #if the block is not in the first layer
            #projection matrix from prev layer as from Eq. 1
            self.U = SlimConv2d(ch_in, ch_out, 
                                kernel, stride, padding, 
                                activation_fn=None,
                                initializer=initializer) 
            
            self.V.extend([SlimConv2d(ch_in, ch_in, 
                                      kernel=1, 
                                      stride=1, 
                                      padding=0,
                                      activation_fn=None,
                                      initializer=initializer) 
                           for _ in range(col)])
            #NOTE: its not clear from paper if alpha is parameter vector with `ch_in` learnable
            # values or a single scalar for complete MLP adapter
            # self.alphas.extend([nn.Parameter(torch.tensor(np.random.choice([1, .1, .001])), requires_grad=True) 
            #                    for _ in range(col)])
            self.alphas.extend([nn.Parameter(torch.tensor(alpha), requires_grad=True) 
                               for _ in range(col)])


    def forward(self, inputs):
        """this forward solution uses adapters with non-linear mapping and learned scalars
        as proposed by orig paper, eq. 2; instead of linear mapping of eq. 1.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.W(inputs[-1])
        prev_columns_out = [v( a * x ) for v,a,x in zip(self.V, self.alphas, inputs)]
        if prev_columns_out:
            return self.activation(cur_column_out + self.U(self.activation(sum(prev_columns_out))))
        else:
            return self.activation(cur_column_out)
        
        #return self.activation(cur_column_out + adapters_out)

    # def forward(self, inputs):
    #     """this forward solution uses the original lateral connections as 
    #     proposed by orig paper, eq. 1 (no non-linear or learned scalars).
    #     """
    #     if not isinstance(inputs, list):
    #         inputs = [inputs]
    #     cur_column_out = self.W(inputs[-1])
    #     prev_columns_out = [mod(x) for mod, x in zip(self.V, inputs)]
    #     return self.activation(cur_column_out + self.U(sum(prev_columns_out)))


class PNN(TorchModelV2, nn.Module):
    '''Progressive Neural Network proposed by https://arxiv.org/abs/1606.04671.
    The base Torch Module is extended by some members that allow to add new tasks to the network
    by adding `columns` to the current structure. Each column replicates the base architecture
    and also adds `adapter layers` from ALL old columns to the new ones for positive forward
    transfer.
    Note: the input size (observation space) is required to stay the same for ALL the tasks. 
    The reason is that for the proposed positive forward transfer, the feature spaces from 
    each previous task must be propagated to the currently used column.
    '''
    def __init__(self, obs_space, action_space, num_outputs, 
                 model_config, name, **custom_model_config):
        # this is only given because its required by the V2 model, the fields 
        #   `obs_space, action_space, num_outputs`
        # are overridden with each new task added to the PNN
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # mode of the PNN
        self.training = custom_model_config.get('in_training', True)
        
        # Conv Blocks
        self.conv_specs = custom_model_config.get('conv_filters', [])
        self.conv_activations = custom_model_config.get('conv_activation')
        # FC Blocks
        self.fc_specs = custom_model_config.get('fcnet_hiddens', [])
        self.fc_activations = custom_model_config.get('fcnet_activation')
        assert self.conv_specs or self.fc_specs, \
            "PNN must have at least one hidden layer for lateral connections!"
        
        try:    
            #catch cases where experimentation HP search is not resolved, e.g. when
            # a raw Trainer obj is instantiated from a tune config 
            self.alpha_inits = float(custom_model_config.get('alpha_value', None))
        except TypeError:
            self.alpha_inits = None

        # the PNN holds a list of every task input / output added to it in the same
        # order the tasks were added
        self.task_specs = []
        # input layer shape of the first column, all consecutive tasks must have the same shape;
        # set when adding the first task
        self.INIT_SIZE = None

        self.columns = nn.ModuleList([])
        # the column id of the current task. Since the `input_dict` of rllib does not support 
        # additional params the task/column to run must be set by an extra function
        self._cur_task_id = None
        self._active_cols = 0    # the number of active cols for the current task, typ. `_cur_task_id + 1`
        self.n_hiddens = None    # number of total hidden layers 
        self._cur_value = None   # current value network output

        # init the PNN if checkpoint is given in the config
        ckpt_file = custom_model_config.get('checkpoint','')
        retrain_last_col = custom_model_config.get('retrain_last_col',False)
        if ckpt_file and os.path.isfile(ckpt_file):
            self._init_model(checkpoint=ckpt_file, 
                             env_specs=custom_model_config.get('build_columns'),
                             retrain_last_col=retrain_last_col)

        if self.training:
            # if PNN has at least one column and it shall be retrained, skip add next column
            # else add the next column to the PNN
            if not (self.columns and retrain_last_col):
                self.next_task(obs_space, action_space)


    @property
    def active_cols(self):
        return self._active_cols

    @property
    def cur_task_id(self):
        return self._cur_task_id

    def _init_model(self, checkpoint, env_specs=None, retrain_last_col=False):
        """Init the PNN with a given checkpoint and heads from env_specs.
        IF `env_specs` is given, it needs to be a list of env spec dicts 
        for each task or path to a env spec dump. If missing, the spec is 
        tried to be loaded from a specific file in the `checkpoint` dir.
        If `retrain_last_col` is True, the last column to be initialized will
        also be retrained instead of a new column added for training.
        """
        import re
        if type(env_specs) not in (list, tuple):
            # env spec not given via config or not correct type
            if type(env_specs)==str and os.path.isfile(env_specs):
                env_spec_path = env_specs
            else:
                ckpt_dir = os.path.dirname(checkpoint)
                env_spec_path = os.path.join(ckpt_dir, ENV_SPEC_SAVE_FILE)
            # try to load env_spec pickle file from checkpoint location
            if os.path.isfile(env_spec_path):
                with open(env_spec_path, 'rb') as f:
                    env_specs = pickle.load(f)
            else:
                raise ValueError('Error initializing the PNN: checkpoint to restore PNN from was '
                                'given but no env_spec is available either via custom_model_config[build_columns] '
                                f'or a pickle file in the checkpoint dir ({env_spec_path}).')
        
        if all(["task_id" in spec.keys() for spec in env_specs]):
            # restore ordering of tasks if possible
            env_specs = sorted(env_specs, key=lambda x: x["task_id"])

        #check the matching between task checkpoint number and env spec length if possible
        #TODO: this is sub-optimal handling of env_specs because it relys on correct "T-x" number in the checkpoint
        # to correctly cut the env_spec down...
        # Better: dont check any ckpt naming convention and always create given env_spec column (needs rework of 
        # whole env_spec handling here and in the mains) 
        try:
            taskStr = re.findall(r'T[s]?[-]?[0-9]+',os.path.basename(checkpoint))[0]
            task_id = int(re.findall(r'[0-9]+',taskStr)[0])
        except:
            #no task number could be found in the ckpt name: assume to restore all columns in env_specs
            pass
        else:
            if task_id >= len(env_specs):
                raise ValueError('Error initializing the PNN: The checkpoint to be loaded implies '
                                f'{task_id+1} tasks but the env spec list has only {len(env_specs)} '
                                'task specs!')
            env_specs = env_specs[:task_id+1]

        for c,spec in enumerate(env_specs):
            obs, act = spec['observation_space'], spec['action_space']
            self.next_task(obs, act)
        self.import_from_h5(checkpoint)
        #freeze all previous columns to prevent retraining them except
        # the last if retrain_last_col is true
        self.freeze_columns(skip= [c] if retrain_last_col else None)


    def next_task(self, obs_space, action_space):
        """Adds a new column to the PNN given the observation and action
        space of the new task. Sets also automatically the current task id 
        to the this task.
        """
        if not self.columns:
            # first task (column) is added to PNN
            # init the obs shape of the original first PNN column 
            self.INIT_SIZE = self._in_shape_transformer(obs_space)
        else: 
            assert self.INIT_SIZE == self._in_shape_transformer(obs_space), \
                "Observation space (input shape={}) of the new ".format(obs_space.shape)+ \
                "task does not match the input shape of the PNN: {}".format(self.INIT_SIZE)

        next_task_id = len(self.columns)
        
        #get the output shape depending on the env action space
        _, num_outputs = get_action_dist(action_space, self.model_config)

        column = nn.ModuleList()
        #hidden layers:
        prev_layer_size = self.INIT_SIZE
        l_depth = 0
        # Conv hiddens
        for spec in self.conv_specs[:-1]:
            (w, h, ch_in) = prev_layer_size
            size_in = [w, h]
            ch_out = spec[0]
            kernel = spec[1]
            stride = [spec[2]]*2 if isinstance(spec[2], int) else spec[2]
            padding, size_out = same_padding(size_in, kernel, stride)
            activation = get_activation_fn(self.conv_activations, framework='torch')

            pnn_block = PNNConvBlock(next_task_id, l_depth, 
                                     ch_in, ch_out, 
                                     kernel, stride, padding, 
                                     activation(),
                                     alpha=self.alpha_inits)
            
            column.append(pnn_block)
            prev_layer_size = list(map(int, size_out)) + [ch_out]
            l_depth += 1
        # Conv to FC conversion
        if self.conv_specs:
            (h, w, ch_in) = prev_layer_size
            size_in = [h, w]
            ch_out = self.conv_specs[-1][0]
            kernel = self.conv_specs[-1][1]
            stride = [self.conv_specs[-1][2]]*2 \
                if isinstance(self.conv_specs[-1][2], int) else self.conv_specs[-1][2]
            _, size_out = same_padding(size_in, kernel, stride)
            activation = get_activation_fn(self.conv_activations, framework='torch')

            pnn_block = nn.Sequential(
                            PNNConvBlock(next_task_id, l_depth, 
                                        ch_in, ch_out, 
                                        kernel, stride, None, 
                                        activation(),
                                        alpha=self.alpha_inits),
                            nn.Flatten()
                        )
            
            column.append(pnn_block)
            size_out = np.array(size_out) - np.array(kernel) + 1
            prev_layer_size = int(np.prod(size_out) * ch_out)
            l_depth += 1
        # FC hiddens
        for spec in self.fc_specs:
            out_shape = spec
            activation = get_activation_fn(self.fc_activations, framework='torch')
            pnn_block = PNNLinearBlock(next_task_id, l_depth, 
                                       prev_layer_size, out_shape, 
                                       activation(), alpha=self.alpha_inits)

            column.append(pnn_block)
            prev_layer_size = out_shape
            l_depth += 1
        
        # output (logits and value) layer
        _logits = PNNLinearBlock(next_task_id, l_depth, prev_layer_size, num_outputs,
                                 activation=None, initializer=nn.init.xavier_uniform_,
                                 alpha=self.alpha_inits)
        #shared value layer, if value nn shall be independent build complete parallel 
        #NN and use model_config.get("vf_share_layers") (bool) in ctor
        _value  = PNNLinearBlock(next_task_id, l_depth, prev_layer_size, 1, 
                                 activation=None, initializer=normc_initializer,
                                 alpha=self.alpha_inits)    
        column.append(nn.ModuleList([_logits, _value]))

        self.columns.append(column)
        self.n_hiddens = l_depth
        self.set_task_id(next_task_id)

        #rewrite the task-specific fields of the ModelV2 parent
        #NOTE: this is not used by any parent, but may be used by the callers (agent, etc)
        self.num_outputs = num_outputs
        self.obs_space = obs_space
        self.action_space = action_space
        self._cur_value = None

        self.task_specs.append({
                "task_id": next_task_id,    #this corresponds to the list index
                "observation_space": obs_space,
                "action_space": action_space,
                "num_outputs": num_outputs
            })

        #verify that the new column is on the same device as the PNN object
        #NOTE: this could cause issues when training on multiple GPUs
        # _col_alloc = str(next(self.columns[-1].parameters()).device)
        # if not _col_alloc == getattr(self, 'on_device', _col_alloc):
        #     print("Added column is not on the device assigned to the PNN! "+ \
        #     "Is on {}; sould be on {}. Try to move Module...".format(_col_alloc, self.on_device))
        #     self.to(torch.device(getattr(self, 'on_device', _col_alloc)))

        # if self.use_cuda:
        #     self.cuda()


    def forward(self, input_dict, state, seq_lens):
        assert self.columns, \
            'PNN should at least have one column (missing call to `next_task` ?)'
        assert self._cur_task_id is not None, \
            "No task (id) set for forward pass (missing call to `set_task_id`?)"
        
        #use the flattened observations if no conv layers are used
        if self.conv_specs:
            # switch to channel-major since Torch Conv layers use input shapes: (B,C​,H​,W)
            x = input_dict["obs"].float().permute(0, 3, 1, 2)
        else:
            x = input_dict["obs_flat"].float()
        
        inputs = [x for _ in range(self._active_cols)]  
        for l in range(0, self.n_hiddens):
            outputs = []
            for c in range(self._active_cols):
                outputs.append(self.columns[c][l](inputs[:c+1]))
            inputs = outputs
        
        #output heads
        logits, values = [], []
        for c in range(self._active_cols):
            logits.append(self.columns[c][l+1][0](outputs[:c+1]))
            values.append(self.columns[c][l+1][1](outputs[:c+1]))

        self._cur_value = values[self._cur_task_id].squeeze(1)

        return logits[self._cur_task_id], state

    
    def set_task_id(self, task_id):
        """Set the PNN forward path to the given task (column) ID.
        The given task_id can be used like a python list indice (-1 is valid).
        """
        assert self.columns, 'No columns in the PNN to set the task id to (missing call to `next_task` ?)'
        if isinstance(task_id, torch.Tensor):
            task_id = int(task_id.item())
        assert task_id < len(self.columns), 'Requested task id is outside the column range of the PNN'
        self._cur_task_id = task_id
        #nr of columns needed to generate output for `task_id`
        self._active_cols = len(self.columns[:task_id])+1     #handle case `task_id=-1`


    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


    def freeze_columns(self, skip=None):
        """Freezes all PNN columns not in `skip` after training.
        """
        if skip == None:
            skip = []
        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False


    #NOTE: with _all=False default the policy optimizer will always only have the 
    # model parameters of the last column (see rllib/policy/torch_policy.py#L372)
    @override(nn.Module)
    def parameters(self, _all=False):
        if _all or not self.columns:
            return nn.Module.parameters(self)
        return self.columns[self._cur_task_id].parameters()


    @override(TorchModelV2)
    def import_from_h5(self, h5_file):
        """Imports weights from an checkpoint file and the 
        task spec list from the same dir if available.
        This can be used when the model was checkpointed with
        `export_checkpoint`.
        Args:
            h5_file (str): The h5 file name to import weights from.
        Example:
            >>> trainer = MyTrainer()
            >>> trainer.import_policy_model_from_h5("/tmp/model_ckpt.h5")
            >>> for _ in range(10):
            >>>     trainer.train()
        """
        #This should only set the model parameters
        #check the device the model is located on
        dev = next(self.parameters()).device
        checkpoint = torch.load(h5_file, map_location=dev)
        try:
            state_dict = checkpoint["model_state_dict"]
        except KeyError:
            state_dict = checkpoint
        
        # ckpt_dir = os.path.dirname(h5_file)
        # env_spec_path = os.path.join(ckpt_dir, ENV_SPEC_SAVE_FILE)
        # if os.path.isfile(env_spec_path):
        #     with open(env_spec_path, 'rb') as f:
        #         env_specs = pickle.load(f)
        #     self.task_specs = env_specs
                    
        return self.load_state_dict(state_dict, strict=True)
    
    #TODO: check if trainer.train() / evaluate() correctly call the torch model.train() / .eval() fns to set the model in the correct state after creation AND loading
    # must be somewhere in trainer, sampler, policy...

    def export_checkpoint(self, export_dir, filename):
        """ This is an additional store method for the model parameters only.
        In Rllib framework one can normally call the restore function of the
        trainer (`export_policy_checkpoint`) instead of this one.
        Example:
            >>> trainer = MyTrainer()
            >>> for _ in range(10):
            >>>     trainer.train()
            >>> trainer.workers.local_worker().for_policy(
            >>>     lambda p: p.model.export_checkpoint("/tmp","weights.h5"))
        """
        #This solution works because state_dict() reads self._parameters instead of 
        # the overridden parameters() function.
        try:
            os.makedirs(export_dir)
        except OSError as e:
            # ignore error if export dir already exists
            if e.errno != errno.EEXIST:
                raise
        export_path = os.path.join(export_dir, filename)
        save_dict = {'model_state_dict': self.state_dict()}
        torch.save(save_dict, export_path)
        # save the task_specs which are needed for the PNN to build
        # the correct input/output heads for each task
        with open(os.path.join(export_dir, ENV_SPEC_SAVE_FILE), 'wb') as f:
            pickle.dump(self.task_specs, f)


    def export_model(self, export_dir):
        """Function for complete model export. This will save the whole 
        model BUT is project dependent and breaks if model class / 
        project structure is changed!
        Example:
            >>> trainer = MyTrainer()
            >>> for _ in range(10):
            >>>     trainer.train()
            >>> trainer.export_policy_model("/tmp/")
            >>> #OR
            >>> trainer.workers.local_worker().for_policy(
            >>>     lambda p: p.model.export_model("/tmp/"))
        Load:
            >>> # Model class must be defined somewhere
            >>> model = torch.load("/tmp/model.pth")
        """
        #No lambdas are allowed in the model when using this bc Troch uses
        #pickel to save the module which doesnt support lambdas
        export_file = os.path.join(export_dir, "model_Ts{}.pth".format(len(self.columns)-1))
        torch.save(self, export_file)
        # save the task_specs which are needed for the PNN to build
        # the correct input/output heads for each task
        with open(os.path.join(export_dir, ENV_SPEC_SAVE_FILE), 'wb') as f:
            pickle.dump(self.task_specs, f)
    

    def summary(self):
        '''Generates a summary string similar to Keras model summary.
        Requires torch-summary package (`pip install torchsummary`).
        '''
        try:
            from common.torchsummary import summary_string
        except ImportError:
            ret_str = 'Could not import torch-summary function '\
                      'from `common.torchsummary`!'
            return ret_str, (None, None)
        
        summary_str, (total_params, trainable_params) = \
            summary_string(self, input_size=self.INIT_SIZE, 
                           device=next(self.parameters()).device)
        
        return summary_str, (total_params, trainable_params)
        

    # @override(nn.Module)
    # def to(self, *args, **kwargs):
    #     """This overrides the Torch.nn.Module.to() method to also track
    #     the device allocation.
    #     """
    #     _self = nn.Module.to(self, *args, **kwargs)
    #     if not hasattr(self, 'on_device'):
    #         self.on_device = str(self.__alloc_param.device)
    #         # del self.__alloc_param
    #     return _self


    def _in_shape_transformer(self, obs):
        if self.conv_specs:
            return obs.shape
        elif self.fc_specs:
            #NOTE: for Atari deepmind frames (84x84x4) this will go up to 28224 Neurons
            return int(np.product(obs.shape))
        else:
            return obs.shape

    #Device allocation is handeled by Policy ctor
    # def cuda(self, *args, **kwargs):
    #     self.use_cuda = True
    #     super(PNN, self).cuda(*args, **kwargs)

    # def cpu(self):
    #     self.use_cuda = False
    #     super(PNN, self).cpu()

