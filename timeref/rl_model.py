"""
RL environment for time-resolved fitting
"""

import logging
import numpy as np
import gymnasium as gym

from refl1d.names import QProbe, Parameter, Experiment

from . import model_utils


class SLDEnv(gym.Env):
    def __init__(
        self,
        initial_state_file=None,
        final_state_file=None,
        data=None,
        reverse=True,
        allow_mixing=False,
        find_scale=False,
        mix_first_action=False,
        use_steady_states=True,
    ):
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
        self.mix_first_action = mix_first_action
        self.use_steady_states = use_steady_states
        self.find_scale = find_scale
        self.time_stamp = len(self.data) - 1 if self.reverse else 0
        self.start_state = True

        if data is None:
            self.q = np.logspace(np.log10(0.009), np.log10(0.2), num=150)
        else:
            self.q = self.data[0][0]

        # Set up the model
        self.setup_model()

        # The state will correspond to the [time interval i] / [number of time intervals]
        self.time_stamp = len(data) - 1 if self.reverse else 0
        self.time_increment = -1 if self.reverse else 1
        self.start_state = True

        # Determine action space, normalized between 0 and 1
        action_size = len(self.low_array)
        if allow_mixing:
            action_size += 1
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=[action_size], dtype=np.float32
        )
        # Observation space is the timestamp
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def check_data(self, data):
        data_list = []
        for d in data:
            _d = np.asarray(d)
            data_list.append(np.asarray(d))
        return data_list

    def determine_scale(self, expt: Experiment) -> float:
        """
        Determine scale factor from first data point
        """
        # Check if time_stamp is valid for the data
        if self.time_stamp >= len(self.data):
            # If time_stamp is out of range, use the last available data point
            time_idx = len(self.data) - 1
        else:
            time_idx = self.time_stamp
        time_data = self.data[time_idx]

        dq = self.q_resolution * self.q
        probe = QProbe(self.q, dq)

        probe.intensity = Parameter(value=1.0, name=expt.probe.intensity.name)
        ref_model = Experiment(probe=probe, sample=expt.sample)
        _, refl = ref_model.reflectivity()

        scale = np.sum(time_data[1]) / np.sum(refl)
        return scale

    def setup_model(self):
        # Create QProbe matching tNR data
        dq = self.q_resolution * self.q
        probe = QProbe(self.q, dq)

        expt = model_utils.expt_from_json_file(self.expt_file)

        scale_value = expt.probe.intensity.value

        if self.find_scale:
            scale_value = self.determine_scale(expt)
            logging.debug(f"Determined scale factor: {scale_value}")

        # Set initial intensity and background to those in the experiment
        probe.intensity = Parameter(value=scale_value, name=expt.probe.intensity.name)
        probe.background = Parameter(
            value=expt.probe.background.value, name=expt.probe.background.name
        )
        self.ref_model = Experiment(probe=probe, sample=expt.sample)

        if self.end_expt_file:
            self.end_model = model_utils.expt_from_json_file(self.end_expt_file)
        else:
            self.end_model = None
        # May need to set the q resolution
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
                self.low_array.append(layer.thickness.bounds[0])
                self.high_array.append(layer.thickness.bounds[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].thickness.value)
            if not layer.interface.fixed:
                self.par_labels.append(str(layer.interface))
                self.parameters.append(layer.interface.value)
                self.low_array.append(layer.interface.bounds[0])
                self.high_array.append(layer.interface.bounds[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].interface.value)
            if not layer.material.rho.fixed:
                self.par_labels.append(str(layer.material.rho))
                self.parameters.append(layer.material.rho.value)
                self.low_array.append(layer.material.rho.bounds[0])
                self.high_array.append(layer.material.rho.bounds[1])
                if self.end_model:
                    self.end_parameters.append(
                        self.end_model.sample[i].material.rho.value
                    )
            if not layer.material.irho.fixed:
                self.par_labels.append(str(layer.material.irho))
                self.parameters.append(layer.material.irho.value)
                self.low_array.append(layer.material.irho.bounds[0])
                self.high_array.append(layer.material.irho.bounds[1])
                if self.end_model:
                    self.end_parameters.append(
                        self.end_model.sample[i].material.irho.value
                    )
        self.parameters = np.asarray(self.parameters)
        self.end_parameters = np.asarray(self.end_parameters)
        self.low_array = np.asarray(self.low_array)
        self.high_array = np.asarray(self.high_array)
        self.normalized_parameters = (
            2 * (self.parameters - self.low_array) / (self.high_array - self.low_array)
            - 1
        )
        # TODO: the end state might not have the same ranges
        if self.end_model:
            self.normalized_end_parameters = (
                2
                * (self.end_parameters - self.low_array)
                / (self.high_array - self.low_array)
                - 1
            )

    def convert_action_to_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Convert parameters from action space to physics spaces
        """
        deltas = self.high_array - self.low_array
        return self.low_array + deltas * (parameters + 1.0) / 2.0

    def convert_action_uncertainties_to_parameters(
        self, uncertainties: np.ndarray
    ) -> np.ndarray:
        """
        Convert uncertainty on parameters from action space to physics spaces
        """
        deltas = self.high_array - self.low_array
        return deltas * uncertainties / 2.0

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
            mixing = (action[-1] + 1) / 2.0
            action = action[:-1]
        truncated = False
        info = {}
        pars = self.convert_action_to_parameters(action)

        self.q = self.data[self.time_stamp][0]
        # Should check Q resolution
        if len(self.data[self.time_stamp]) > 3:
            dq = self.data[self.time_stamp][3] / 2.35
        else:
            dq = self.q_resolution * self.data[self.time_stamp][0]
        probe = QProbe(self.q, dq, data=None)
        probe.intensity = Parameter(
            value=self.ref_model.probe.intensity.value,
            name=self.ref_model.probe.intensity.name,
        )
        probe.background = Parameter(
            value=self.ref_model.probe.background.value,
            name=self.ref_model.probe.intensity.name,
        )
        self.ref_model.probe = probe
        self.set_model_parameters(pars)
        _, self.refl = self.ref_model.reflectivity()

        if self.allow_mixing and not self.start_state:
            if self.mix_first_action:
                self.set_model_parameters(self.first_time_pars)
                _, _refl = self.ref_model.reflectivity()
            else:
                if self.reverse:
                    self.set_model_parameters(self.end_parameters)
                else:
                    self.set_model_parameters(self.parameters)
                _, _refl = self.ref_model.reflectivity()
            self.refl = (1 - mixing) * self.refl + mixing * _refl
        elif self.allow_mixing and self.start_state:
            self.first_time_pars = np.copy(pars)

        # Compute reward
        idx = self.data[self.time_stamp][2] > 0
        reward = -np.sum(
            (self.refl[idx] - self.data[self.time_stamp][1][idx]) ** 2
            / self.data[self.time_stamp][2][idx] ** 2
        ) / len(self.data[self.time_stamp][2][idx])

        # Store the chi2
        self.chi2 = -reward

        if self.reverse:
            terminated = self.time_stamp <= 0
        else:
            terminated = self.time_stamp >= len(self.data) - 1

        # Move to the next time time_stamp
        self.time_stamp += self.time_increment
        state = self.time_stamp / (len(self.data) - 1)
        state = np.array([state], dtype=np.float32)

        # Maybe add a term to minimze the change in parameters

        # Add a term for the boundary conditions (first and last times)
        if self.use_steady_states:
            if self.start_state:
                reward -= len(self.data) * np.mean(
                    (action - self.normalized_parameters) ** 2
                )

            if terminated and self.end_model and not self.allow_mixing:
                reward -= len(self.data) * np.mean(
                    (action - self.normalized_end_parameters) ** 2
                )

        self.start_state = False

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_stamp = len(self.data) - 1 if self.reverse else 0
        self.setup_model()
        state = self.time_stamp / (len(self.data) - 1)
        state = np.array([state], dtype=np.float32)
        self.start_state = True
        info = {}
        return state, info

    def render(self, action=0, reward=0):
        print(action)

    def plot(self, scale=1, newfig=True, errors=False, label=None):
        """
        Plot the current state of the SLD environment.

        This is a wrapper around the plotting function in timeref.reports.plotting
        to maintain API compatibility while separating concerns.
        """
        from .reports.plotting import plot_sld_env_state

        logging.warning(
            "SLDEnv.plot is deprecated. Use the plotting function from timeref.reports.plotting instead."
        )
        plot_sld_env_state(self, scale=scale, newfig=newfig, errors=errors, label=label)
