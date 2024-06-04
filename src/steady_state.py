"""
    RL environment for steady-state fitting
"""
import numpy as np
import gymnasium as gym

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

import json
import refl1d
from refl1d.names import *
from bumps.fitters import fit

import fitting.model_utils

MIN_VALUES = [0, 0, -5, 0]
MAX_VALUES = [1000, 200, 10, 10]
SPACE_AS_TUPLE = False
VARY_IRHO = False

class AnalyzerEnv(gym.Env):

    def __init__(self, initial_state: list, data: np.array, max_layers: int=10, engine='amoeba'):
        """
            Initial and final states are in chronological order. The reverse parameter
            will take care of swapping the start and end states.
        """
        super().__init__()

        # Fitting engine
        self.engine = engine

        # Initial guess to start with as our initial state
        self.initial_state = initial_state

        # Data to fit
        self.data = data

        # Maximum number of layers
        self.max_layers = max_layers

        # Reward cutoff
        self.reward_cutoff = 1.5

        # Action space
        # -1: delete, 0: nothing, 1: insert
        # layer index [int from 0 to 1]
        # layer parameter [0:thickness, 1:interface, 2:rho, 3:irho]
        # value [float between -1 and 1]
        if SPACE_AS_TUPLE:
            self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3, start=-1),
                                                  gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                                                  gym.spaces.Discrete(4),
                                                  gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)))
            self.observation_space = gym.spaces.Tuple((gym.spaces.Box(0, self.max_layers, shape=(1,), dtype=np.int32),
                                                       gym.spaces.Box(0, 1, shape=(self.max_layers, 4), dtype=np.float32)))
        else:
            self.action_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32)
            n_obs = self.max_layers * 4 + 1
            self.observation_space = gym.spaces.Box(0, 1, shape=(n_obs,), dtype=np.float32)

    def parameters_to_observation(self, state):
        """
            Convert the state to the observation space.
            The state is a list of layers, each with four parameters.
        """
        observation = np.zeros_like(self.observation_space.sample())

        if SPACE_AS_TUPLE:
            observation[0][0] = len(state)
            for i, layer in enumerate(state):
                for j, param in enumerate(layer):
                    observation[1][i, j] = (param - MIN_VALUES[j]) / (MAX_VALUES[j] - MIN_VALUES[j])
        else:
            observation[0] = len(state) / self.max_layers
            for i, layer in enumerate(state):
                for j, param in enumerate(layer):
                    observation[i * 4 + j + 1] = (param - MIN_VALUES[j]) / (MAX_VALUES[j] - MIN_VALUES[j])
        
        return observation

    def _update_state_from_tuple(self, action, verbose=False):
        """
            Update the state based on the action
        """
        action_type = action[0]            
        layer_param = action[2]
        layer_index = int(np.floor(action[1] * (self.state[0][0] - 2.00001)) + 1)

        # TODO: check edge cases
        if action_type == -1:
            # Delete layer
            new_state = np.delete(self.state[1], layer_index, axis=0)
            self.state = [self.state[0] - 1, new_state]
            #print('Deleting layer', layer_index)
        elif action_type == 0:
            # Keep structure as-is, but modify parameters
            self.state[1][layer_index, layer_param] = action[3]
            #print("Setting layer %d, parameter %d to %f" % (layer_index, action[2], action[3]))
        elif action_type == 1:
            # Insert layer and set layer parameters
            layer_pars = np.asarray([0.05, 0.005, 0.6, 0])
            layer_pars[layer_param] = action[3]
            # When inserting a layer, we refer to the following index, therefore the +1
            new_state = np.insert(self.state[1], layer_index + 1, layer_pars, axis=0)
            self.state = [self.state[0] + 1, new_state]
            #print("Inserting layer at index %d with parameters %s=%s" % (layer_index, action[2], layer_pars))

    def _update_state_from_array(self, action, verbose=False):
        """
            Update the state based on the action

            TODO: give less weight to adding/removing layers

        """
        n_layers = int(np.floor(self.state[0] * (self.max_layers - 0.00001) ) + 1)
        action_type = int(np.floor(action[0] * 2.99999) - 1)
        n_max = 3.99999 if VARY_IRHO else 2.99999
        layer_param = int(np.floor(action[2] * n_max))
        

        if action_type == -1:
            # Delete layer
            # First check whether we have more than the buffer and substrate,
            # otherwise we cannot delete any layers
            if n_layers < 3:
                return -1
            # Decrease the number of layers
            self.state[0] = (n_layers - 1) / self.max_layers
            # Remove the parameters for the deleted layer
            # The layer cannot be the buffer or the substrate
            layer_index = int(np.floor(action[1] * (n_layers - 2.00001))) + 1
            pars_to_remove = list(range(layer_index * 4 + 1, layer_index * 4 + 5))
            self.state = np.delete(self.state, pars_to_remove)
            # Pad the end of the observation array
            self.state = np.insert(self.state, len(self.state), np.zeros(4))
            if verbose:
                print(' - Deleting layer', layer_index)
        elif action_type == 0:
            # Keep structure as-is, but modify parameters
            # The layer can be the buffer but not the substrate
            layer_index = int(np.floor(action[1] * (n_layers-1.00001)))
            self.state[layer_index * 4 + layer_param + 1] = action[3]
            if verbose:
                print(" - Setting layer %d, parameter %d to %f" % (layer_index, action[2], action[3]))
        elif action_type == 1:
            # First check whether we have reached the maximum number of layers
            if n_layers >= self.max_layers:
                return -1
            # Insert layer and set layer parameters
            layer_pars = np.asarray([0.05, 0.005, 0.6, 0])
            layer_pars[layer_param] = action[3]
            # Increase the number of layers
            self.state[0] = (n_layers + 1) / self.max_layers
            # When inserting a layer, we refer to the following index, therefore the +1
            layer_index = int(np.floor(action[1] * (n_layers - 1.00001)))
            # The layer can be the buffer but not the substrate
            self.state = np.insert(self.state, (layer_index + 1) * 4 + 1, layer_pars)
            # Remove the last layer to keep the array of the same langth
            self.state = np.delete(self.state, [-1, -2, -3, -4])
            if verbose:
                print(" - Inserting layer at index %d with parameters %s=%s" % (layer_index+1, action[2], layer_pars))

        return 0

    def update_state(self, action, verbose=False):
        """
            Update the state based on the action
        """
        if SPACE_AS_TUPLE:
            return self._update_state_from_tuple(action, verbose=verbose)
        else:
            return self._update_state_from_array(action, verbose=verbose)

    def print_parameters(self, pars):
        print("\nCurrent parameters:")
        for i, layer in enumerate(pars):
            print('Layer %d: %g\t%g\t%g\t%g' % (i, layer[0], layer[1], layer[2], layer[3]))
        print()

    def _get_pars_from_state(self):
        """
            Convert the observation space to real units
        """
        # Convert the observation space to real units
        pars = []
        if SPACE_AS_TUPLE:
            for i in range(self.state[0][0]):
                thickness = self.state[1][i, 0] * (MAX_VALUES[0] - MIN_VALUES[0]) + MIN_VALUES[0]
                interface = self.state[1][i, 1] * (MAX_VALUES[1] - MIN_VALUES[1]) + MIN_VALUES[1]
                rho = self.state[1][i, 2] * (MAX_VALUES[2] - MIN_VALUES[2]) + MIN_VALUES[2]
                irho = self.state[1][i, 3] * (MAX_VALUES[3] - MIN_VALUES[3]) + MIN_VALUES[3]
                pars.append([thickness, interface, rho, irho])
        else:
            n_layers = int(np.floor(self.state[0] * (self.max_layers - 0.00001) )) + 1
            for i in range(n_layers):
                thickness = self.state[i * 4 + 1] * (MAX_VALUES[0] - MIN_VALUES[0]) + MIN_VALUES[0]
                interface = self.state[i * 4 + 2] * (MAX_VALUES[1] - MIN_VALUES[1]) + MIN_VALUES[1]
                rho = self.state[i * 4 + 3] * (MAX_VALUES[2] - MIN_VALUES[2]) + MIN_VALUES[2]
                irho = self.state[i * 4 + 4] * (MAX_VALUES[3] - MIN_VALUES[3]) + MIN_VALUES[3]
                pars.append([thickness, interface, rho, irho])
        return pars

    def _get_experiment_from_pars(self, pars):
        """
            Create the reflectivity model based on the parameters
        """
        # Compute the reflectivity
        self.q = self.data[0]
        if len(self.data) > 3:
            dq = self.data[3] / 2.35
        else:
            dq = self.q_resolution * self.data[0]
        probe = QProbe(self.q, dq, data=[self.data[1], self.data[2]])
        probe.intensity = Parameter(value=1.0, name='intensity')
        probe.background = Parameter(value=0, name='background')

        # Create the model
        sample = None
        n_fit_pars = 0
        for i, layer in enumerate(pars):
            material = SLD(name='L%d' % i, rho=layer[2], irho=layer[3])

            slab = Slab(material=material,
                        thickness=layer[0], 
                        interface=layer[1])

            if i > 0 and i < len(pars)-1:
                slab.thickness.range(MIN_VALUES[0], MAX_VALUES[0])
                slab.interface.range(MIN_VALUES[1], MAX_VALUES[1])
                slab.material.rho.range(MIN_VALUES[2], MAX_VALUES[2])
                n_fit_pars += 3
                if VARY_IRHO:
                    slab.material.irho.range(MIN_VALUES[3], MAX_VALUES[3])
                    n_fit_pars += 1

            sample = slab if sample is None else sample | slab
        experiment = Experiment(probe=probe, sample=sample)
        return experiment, n_fit_pars

    def compute_reflectivity_from_state(self):

        pars = self._get_pars_from_state()
        #self.print_parameters(pars)

        # Create the reflectivity model
        experiment, n_fit_pars = self._get_experiment_from_pars(pars)

        # Fit the model
        #TODO: clean this up!
        if n_fit_pars > 0:
            problem = FitProblem(experiment)
            results = fit(problem, method=self.engine, samples=2000, burn=2000, pop=20, verbose=None)
            # Update the state with the fitted parameters
            # The fit results are in the order interface, irho, rho, thickness
            # Our pars are in the order thickness, interface, rho, irho
            n_pars_per_layer = 4 if VARY_IRHO else 3
            for i, par in enumerate(results.x):
                layer_index = i // n_pars_per_layer + 1
                # interface
                if i % n_pars_per_layer == 0:
                    value = (par - MIN_VALUES[0]) / (MAX_VALUES[0] - MIN_VALUES[0])
                    self.state[layer_index * 4 + 1 + 1] = value
                    #pars[layer_idx][1] = par
                # irho if we vary it, otherwise rho
                elif i % n_pars_per_layer == 1:
                    if VARY_IRHO:
                        value = (par - MIN_VALUES[3]) / (MAX_VALUES[3] - MIN_VALUES[3])
                        self.state[layer_index * 4 + 3 + 1] = value
                        #pars[layer_idx][3] = par
                    else:
                        value = (par - MIN_VALUES[2]) / (MAX_VALUES[2] - MIN_VALUES[2])
                        self.state[layer_index * 4 + 2 + 1] = value
                        #pars[layer_idx][2] = par
                # rho if we vary irho, otherwise thickness
                elif i % n_pars_per_layer == 2:
                    if VARY_IRHO:
                        value = (par - MIN_VALUES[2]) / (MAX_VALUES[2] - MIN_VALUES[2])
                        self.state[layer_index * 4 + 2 + 1] = value
                        #pars[layer_idx][2] = par
                    else:
                        value = (par - MIN_VALUES[0]) / (MAX_VALUES[0] - MIN_VALUES[0])
                        self.state[layer_index * 4 + 0 + 1] = value
                        #pars[layer_idx][0] = par
                # if we vary irho, the last parameter is thickness
                elif i % n_pars_per_layer == 3:
                    value = (par - MIN_VALUES[0]) / (MAX_VALUES[0] - MIN_VALUES[0])
                    self.state[layer_index * 4 + 0 + 1] = value
                    #pars[layer_idx][0] = par

        _, self.refl = experiment.reflectivity()
        self.z, self.sld, _ = experiment.smooth_profile()

    def step(self, action, verbose=False):
        """
            Perform an action on the environment
        """
        truncated = False
        info = {}

        retval = self.update_state(action, verbose=verbose)
        # Stop if we have an illegal action
        if retval == -1:
            return self.state, -100, True, True, info

        self.compute_reflectivity_from_state()

        # Compute the chi2
        idx = self.data[2] > 0
        chi2 = np.sum( (self.refl[idx] - self.data[1][idx])**2 / self.data[2][idx]**2 ) / len(self.data[2][idx])

        # Compute the reward
        # TODO: Add reward term for when the roughness is larger than the thickness

        # We want to reward a decrease in chi2 compared to our initial guess,
        # but also a decrease compared to the previous step
        #reward = self.chi2_0 - chi2
        #reward += self.chi2 - chi2
        reward = - chi2

        self.chi2 = chi2

        self.step_counter = self.step_counter + 1 if reward < 0 else 0

        # Penalize the number of layers above the initial number
        n_layers = int(np.floor(self.state[0] * (self.max_layers - 0.00001) ))
        #print(n_layers, len(self.initial_state))
        added_layers = max(0, n_layers - len(self.initial_state))
        reward -= added_layers * 0.1
        #reward -= self.step_counter * 0.01

        terminated = self.chi2 < self.reward_cutoff
        reward = reward + 100 if terminated else reward

        if self.step_counter > 50:
            if verbose:
                print('Terminating due to too many steps')
            terminated = True
            truncated = True
            reward -= 100

        if verbose:
            self.print_parameters(self._get_pars_from_state())

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Keep track of the state since the action modifies it and we will need it later.
        self.state = self.parameters_to_observation(self.initial_state)
        self.compute_reflectivity_from_state()
        
        idx = self.data[2] > 0        
        self.chi2 = np.sum( (self.refl[idx] - self.data[1][idx])**2 / self.data[2][idx]**2 ) / len(self.data[2][idx])
        self.chi2_0 = self.chi2
        self.step_counter = 0
        info = {}
        return self.state, info

    def render(self, action=0, reward=0):
        print(action)

    def plot(self, scale=1, newfig=True, errors=False, label=None):
        """
        Plot the reflectivity and scattering length density profiles.

        TODO: plot uncertainties on SLD plot

        Args:
            scale (float): Scaling factor for the reflectivity data.
            newfig (bool): Whether to create a new figure for the plot.
            errors (bool): Whether to plot error bars.
            label (str): Label for the data being plotted.

        Returns:
            None
        """
        if newfig:
            fig, axs = plt.subplots(2,1, dpi=100, figsize=(6,8), sharex=False)
        
        ax = plt.subplot(2, 1, 1)
        plt.plot(self.q, self.refl*scale, color='gray')

        idx = self.data[1] > self.data[2]
        if label is not None:
            _label = label
        else:
            _label = 'prediction'

        if errors:
            plt.errorbar(self.data[0][idx], self.data[1][idx]*scale,
                         yerr=self.data[2][idx]*scale, label=_label, linestyle='', marker='.')
        else:
            plt.plot(self.data[0][idx], self.data[1][idx]*scale,
                     label=_label)

        plt.gca().legend()
        plt.xlabel('q [$1/\AA$]')
        plt.ylabel('R(q)')
        plt.xscale('log')
        plt.yscale('log')

        ax = plt.subplot(2, 1, 2)
        plt.plot(self.z[-1] - self.z, self.sld, color='gray')
