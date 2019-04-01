import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import StateEntropyReward
import tensorflow as tf
from tensorflow import logging
from utils import rollout, policy
np.random.seed(0)

SUBS=3
bf = 30
maxiter=50
weights = np.diag([2.0, 2.0, 0.3])
T = 100
T_sim = T
J = 4
N = 10
restarts = 2

with tf.Session() as sess:
    env = gym.make('acrobot-continuous-v1')

    # Initial random rollouts to generate a dataset
    X,Y = rollout(env, None, timesteps=T, random=True, SUBS=SUBS)
    for i in range(1,J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = env.observation_space.shape[0]
    control_dim = env.action_space.shape[0]
    print("state dim::", state_dim, "control dim::", control_dim)

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf)

    R = StateEntropyReward(state_dim=state_dim)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R)

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # cur_rew = 0
        # for t in range(0,len(X_new)):
        #     cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        # print('On this episode reward was ', cur_rew)

        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)