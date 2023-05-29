

import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
print('Main dir:', __MAINDIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)
#NOTE: The RolloutWorkers do not receive the Driver system paths!
print('System Path:\n',sys.path)

import numpy as np
import gym
import cv2

from envs.gym_wrappers import *
import matplotlib.pyplot as plt

#print all the Gym envs
for e in gym.envs.registry.all():
    print(e.id)

baseEnvID = 'PongNoFrameskip-v4'
wrapper = VerticalFlip
# wrapper = lambda env: ZoomAndRescale(env, zoom_factor=1.3)

# env = gym.make(baseEnvID)
# orig_env_state = env.clone_state()
# orig_obs = env.reset()
# print('Original shape: ',orig_obs.shape)
# for i in range(1000):
#     a = np.random.randint(env.action_space.n)
#     orig_obs, r, done, _ = env.step(a)
#     if (i+1) % 500 == 0:
#         plt.imshow(orig_obs)
#         plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#         plt.show()
# orig_env_step_state = env.clone_state()

# print('State at env creation and after stepping equal? {}'.format(orig_env_state==orig_env_step_state))
# print(f'State size (shape): {orig_env_state.shape}')

#----------------------

wrapped_env = wrapper(gym.make(baseEnvID))
wenv_state = wrapped_env.clone_state()
wrapped_obs = wrapped_env.reset()
print('Wrapped shape: ',wrapped_obs.shape)
for i in range(1000):
    a = np.random.randint(wrapped_env.action_space.n)
    wrapped_obs, r, done, _ = wrapped_env.step(a)
    if (i+1) % 500 == 0:
        plt.imshow(wrapped_obs, cmap = 'gray')  #cmap ignored for RGB(A)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
print('Last obs before state save')
plt.imshow(wrapped_obs, cmap = 'gray')  #cmap ignored for RGB(A)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

wenv_step_state = wrapped_env.clone_state()

print('State at wrapped env creation and after stepping equal? {}'.format(wenv_state==wenv_step_state))
print(f'State size (shape): {wenv_state.shape}')


env_2 = gym.make(baseEnvID)
wrapped_env_2 = wrapper(gym.make(baseEnvID))
wrapped_obs_2 = wrapped_env_2.reset()
print('Showing un-restored wrapped env 2')
plt.imshow(wrapped_obs_2, cmap = 'gray')  #cmap ignored for RGB(A)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

wrapped_env_2.restore_state(wenv_step_state)
# wrapped_obs_2_res = wrapped_env_2.unwrapped._get_obs()
wrapped_obs_2_res,_,_,_ = wrapped_env_2.step(0)
print('Showing restored wrapped env 2')
plt.imshow(wrapped_obs_2_res, cmap = 'gray')  #cmap ignored for RGB(A)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

print('image diff between saved state obs and restored obs (smaller is better)')
plt.imshow(wrapped_obs_2_res - wrapped_obs, cmap = 'gray')  #cmap ignored for RGB(A)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

print('DONE')

# cv2.imshow('wrapped Obs', wrapped_obs)
# cv2.waitKey(0)


# cv2.destroyAllWindows()