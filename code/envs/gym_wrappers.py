###################################################
#
# Wrapper collection for the gym environment.
# from OpenAI Baselines https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
#
###################################################
import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

##### Gym general #####


class NormObservation(gym.ObservationWrapper):
    '''Observation wrapper that normalizes the observations
    with a certain feature scaling method.
    Note that this filter can only be used if the boundaries
    of the observation space are the same for all dimensions!
    '''
    def __init__(self, env, mode, **kwargs):
        super(NormObservation,self).__init__(env)
        self.mode = mode
        #check same boundaries for the the R^n space
        if np.all(self.observation_space.high==np.ravel(self.observation_space.high)[0]):
            self.obs_max = self.observation_space.high.max().item()
        else:
            raise ValueError('The upper boundaries of the observation space are not equal for all dims!')
        if np.all(self.observation_space.low==np.ravel(self.observation_space.low)[0]):
            self.obs_min = self.observation_space.low.min().item()
        else:
            raise ValueError('The lower boundaries of the observation space are not equal for all dims!')
        self.min_max_low = kwargs.get('min_max_low',0)
        self.min_max_high = kwargs.get('min_max_high',1)

    def observation(self, obs):
        if self.mode == 'min_max':
            obs = (obs.astype(float) - self.obs_min)/(self.obs_max-self.obs_min)
            obs = obs * self.min_max_high + self.min_max_low
        elif self.mode == 'mean':    #note that this norm results in dynamic ranges
            obs = (obs.astype(float) - np.mean(obs))/(self.obs_max-self.obs_min)
        else:
            raise NotImplementedError('Normalization mode [{}] is not implemented'.format(self.mode))
        return obs


class VerticalFlip(gym.ObservationWrapper):
    """Vertically flip observation"""
    def __init__(self, env=None, **kwargs):
        super(VerticalFlip, self).__init__(env)

    def observation(self, obs):
        return obs[::-1,:,:]


class HorizontalFlip(gym.ObservationWrapper):
    """Horizontally flip observation"""
    def __init__(self, env=None, **kwargs):
        super(HorizontalFlip, self).__init__(env)

    def observation(self, obs):
        return obs[:,::-1,:]


class ZoomAndRescale(gym.ObservationWrapper):
    """Zoom the observation by `zoom_factor` and
    rescale to original size.
    `zoom_factor`s < 1 correspond to a down-scale (zoom-out)
    of the observation (in percent). 
    Values > 1 (zoom-in) result in the edges being cropped off.
    NOTE: this requires the observations to have the channel 
    order: (H,W,C)
    """
    def __init__(self, env=None, **kwargs):
        super(ZoomAndRescale, self).__init__(env)
        self.zoom_factor = kwargs.get('zoom_factor', 0.75)

    def observation(self, obs):
        orig_hw = obs.shape[:2]
        scaled_hw = tuple(map(lambda x: int(float(x) * self.zoom_factor), orig_hw))
        obs = cv2.resize(obs, dsize=scaled_hw[::-1], 
                         interpolation=cv2.INTER_AREA if self.zoom_factor<=1 else cv2.INTER_CUBIC)
        if self.zoom_factor < 1.:
            fill_val = [0] * obs.shape[2]
            d_h, d_w = np.subtract(orig_hw, scaled_hw)
            top, bottom = d_h//2, d_h-(d_h//2)
            left, right = d_w//2, d_w-(d_w//2)
            obs = cv2.copyMakeBorder(obs, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=fill_val)
        elif self.zoom_factor > 1.:
            d_h, d_w = np.subtract(scaled_hw, orig_hw)
            top, bottom = d_h//2, d_h-(d_h//2)
            left, right = d_w//2, d_w-(d_w//2)
            obs = obs[top:-bottom, left:-right, :]

        return obs


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._max_episode_steps and self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

##### Atari specific #####

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, noop_action=0):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0. 
        If env has no NOOP, this wrapper does nothing.
        """
        gym.Wrapper.__init__(self, env)
        self.has_noop = True
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = noop_action
        if env.unwrapped.get_action_meanings()[self.noop_action] != 'NOOP':
            self.has_noop = False


    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset(**kwargs)
        if self.has_noop:
            if self.override_num_noops is not None:
                noops = self.override_num_noops
            else:
                noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
            # assert noops > 0
            obs = None
            for _ in range(noops):
                obs, _, done, _ = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs   #save the last two obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


# # TODO: just a noop wrapper for the actual env to prevent using cv2
# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
