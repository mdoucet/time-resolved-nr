"""
  TODO:
  - Read DREAM fit results into model object
  -
"""
import numpy as np
import gymnasium as gym

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

import json
import refl1d
from refl1d.names import *

import fitting.model_utils


def _dense(x, dtype='d'):
    return np.ascontiguousarray(x, dtype)

def convolve(xi, yi, x, dx, resolution='normal'):
    r"""
    Apply x-dependent gaussian resolution to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    The theory function is a piece-wise linear spline which does not need to
    be uniformly sampled.  The theory calculation points *xi* should be dense
    enough to capture the "wiggle" in the theory function, and should extend
    beyond the ends of the data measurement points *x*. Convolution at the
    tails is truncated and normalized to area of overlap between the resolution
    function in case the theory does not extend far enough.

    *resolution* is 'normal' (default) or 'uniform'. Note that the uniform
    distribution uses the $1-\sigma$ equivalent distribution width which is
    $1/\sqrt{3}$ times the width of the rectangle.
    """
    from refl1d import refllib
    xi, yi, x, dx = _dense(xi), _dense(yi), _dense(x), _dense(dx)
    y = np.empty_like(x)
    if resolution == 'uniform':
        refllib.convolve_uniform(xi, yi, x, dx, y)
    else:
        refllib.convolve_gaussian(xi, yi, x, dx, y)
    return y


class SLDEnv(gym.Env):

    def __init__(self, initial_state_file, final_state_file=None, data=None, reverse=True, allow_mixing=False):
        """
            Initial and final states are in chronological order. The reverse parameter
            will take care of swapping the start and end states.
        """
        super().__init__()
        if reverse:
            self.expt_file = final_state_file
            self.end_expt_file = initial_state_file
        else:
            self.expt_file = initial_state_file
            self.end_expt_file = final_state_file
        self.data = self.check_data(data)
        self.reverse = reverse
        self.q_resolution = 0.028
        self.allow_mixing = allow_mixing

        if data is None:
            self.q = np.logspace(np.log10(0.009), np.log10(0.2), num=150)
        else:
            self.q = self.data[0][0]

        # Set up the model
        self.setup_model()

        # The state will correspond to the [time interval i] / [number of time intervals]
        self.time_stamp = len(data)-1 if self.reverse else 0
        self.time_increment = -1 if self.reverse else 1
        self.start_state = True

        # Determine action space, normalized between 0 and 1
        action_size = len(self.low_array)
        if allow_mixing:
            action_size += 1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[action_size], dtype=np.float32)
        # Observation space is the timestamp
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)

    def check_data(self, data):
        data_list = []
        for d in data:
            _d = np.asarray(d)
            data_list.append(np.asarray(d))
        return data_list

    def setup_model(self):
        self.ref_model = fitting.model_utils.expt_from_json_file(self.expt_file, self.q, q_resolution=self.q_resolution, set_ranges=True)
        if self.end_expt_file:
            self.end_model = fitting.model_utils.expt_from_json_file(self.end_expt_file, self.q, q_resolution=self.q_resolution, set_ranges=True)
        else:
            self.end_model = None
        _, self.refl = self.ref_model.reflectivity()
        self.get_model_parameters()

    def get_model_parameters(self):
        self.par_labels = []
        self.parameters = []
        self.end_parameters = []
        self.low_array = []
        self.high_array = []
        for i, layer in enumerate(self.ref_model.sample):
            if not layer.thickness.fixed:
                self.par_labels.append(str(layer.thickness))
                self.parameters.append(layer.thickness.value)
                self.low_array.append(layer.thickness.bounds.limits[0])
                self.high_array.append(layer.thickness.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].thickness.value)
            if not layer.interface.fixed:
                self.par_labels.append(str(layer.interface))
                self.parameters.append(layer.interface.value)
                self.low_array.append(layer.interface.bounds.limits[0])
                self.high_array.append(layer.interface.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].interface.value)
            if not layer.material.rho.fixed:
                self.par_labels.append(str(layer.material.rho))
                self.parameters.append(layer.material.rho.value)
                self.low_array.append(layer.material.rho.bounds.limits[0])
                self.high_array.append(layer.material.rho.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].material.rho.value)
            if not layer.material.irho.fixed:
                self.par_labels.append(str(layer.material.irho))
                self.parameters.append(layer.material.irho.value)
                self.low_array.append(layer.material.irho.bounds.limits[0])
                self.high_array.append(layer.material.irho.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].material.irho.value)
        self.parameters = np.asarray(self.parameters)
        self.end_parameters = np.asarray(self.end_parameters)
        self.low_array = np.asarray(self.low_array)
        self.high_array = np.asarray(self.high_array)
        self.normalized_parameters = 2 * ( self.parameters - self.low_array ) / (self.high_array - self.low_array) - 1
        #TODO: the end state might not have the same ranges
        if self.end_model:
            self.normalized_end_parameters = 2 * ( self.end_parameters - self.low_array ) / (self.high_array - self.low_array) - 1

    def convert_action_to_parameters(self, parameters):
        """
            Convert parameters from action space to physics spaces
        """
        deltas = self.high_array - self.low_array
        return self.low_array + deltas * (parameters + 1.0) / 2.0

    def set_model_parameters(self, values):
        """
            Parameters are normalized from 0 to 1
        """
        counter = 0

        for i, layer in enumerate(self.ref_model.sample):
            if not layer.thickness.fixed:
                layer.thickness.value = values[counter]
                counter += 1
            if not layer.interface.fixed:
                layer.interface.value = values[counter]
                counter += 1
            if not layer.material.rho.fixed:
                layer.material.rho.value = values[counter]
                counter += 1
            if not layer.material.irho.fixed:
                layer.material.irho.value = values[counter]
                counter += 1
        if not len(values) == counter:
            print("Action length doesn't match model: %s %s" % (len(values), counter))
        self.ref_model.update()

    def step(self, action):
        if self.allow_mixing:
            mixing = (action[-1] + 1)/2.0
            action = action[:-1]
        truncated = False
        info = {}
        pars = self.convert_action_to_parameters(action)

        self.q = self.data[self.time_stamp][0]
        dq = self.q_resolution * self.data[self.time_stamp][0]
        probe = QProbe(self.q, dq, data=None)
        probe.intensity = Parameter(value=self.ref_model.probe.intensity.value,
                                    name=self.ref_model.probe.intensity.name)
        probe.background = Parameter(value=self.ref_model.probe.background.value,
                                    name=self.ref_model.probe.intensity.name)
        self.ref_model.probe = probe
        self.set_model_parameters(pars)
        _, self.refl = self.ref_model.reflectivity()

        if self.allow_mixing:
            self.set_model_parameters(self.parameters)
            _, _refl = self.ref_model.reflectivity()
            self.refl = (1 - mixing) * self.refl + mixing * _refl

        # Compute reward
        idx = self.data[self.time_stamp][2] > 0

        if np.any(np.isnan(self.refl[idx])):
            self.refl = np.ones(len(self.refl))
            print("NaNs in reflectivity")
        reward = -np.sum( (self.refl[idx] - self.data[self.time_stamp][1][idx])**2 / self.data[self.time_stamp][2][idx]**2 ) / len(self.data[self.time_stamp][2][idx])

        # Store the chi2
        self.chi2 = -reward

        if self.reverse:
            terminated = self.time_stamp <= 0
        else:
            terminated = self.time_stamp >= len(self.data)-1

        # Move to the next time time_stamp
        self.time_stamp += self.time_increment
        state = self.time_stamp / (len(self.data)-1)
        state = np.array([state], dtype=np.float32)

        # Add a term for the boundary conditions (first and last times)
        if self.start_state:
            reward -= len(self.data) * np.sum( (action - self.normalized_parameters)**2 ) / len(self.normalized_parameters)
            if self.allow_mixing:
                reward -= 10.*mixing
        if terminated and self.end_model:
            reward -= len(self.data) * np.sum( (action - self.normalized_end_parameters)**2 ) / len(self.normalized_end_parameters)
            reward -= 10.*mixing

        self.start_state = False

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.setup_model()
        self.time_stamp = len(self.data)-1 if self.reverse else 0
        state = self.time_stamp / (len(self.data)-1)
        state = np.array([state], dtype=np.float32)
        self.start_state = True
        info = {}
        return state, info

    def render(self, action=0, reward=0):
        print(action)

    def plot(self, scale=1, newfig=True, errors=False, label=None):
        if newfig:
            fig = plt.figure(dpi=100)
        plt.plot(self.q, self.refl*scale)

        idx = self.data[self.time_stamp][1] > self.data[self.time_stamp][2]
        _label = label if label is not None else str(self.time_stamp)
        if errors:
            plt.errorbar(self.data[self.time_stamp][0][idx], self.data[self.time_stamp][1][idx]*scale,
                         yerr=self.data[self.time_stamp][2][idx]*scale, label=_label)
        else:
            plt.plot(self.data[self.time_stamp][0][idx], self.data[self.time_stamp][1][idx]*scale,
                     label=_label)

        plt.gca().legend()
        plt.xlabel('q [$1/\AA$]')
        plt.ylabel('R(q)')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
