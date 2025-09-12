import os
import json
import numpy as np

import refl1d
from refl1d.names import QProbe, Parameter, SLD, Slab, Experiment

from bumps import serialize

ERR_MIN_ROUGH = 1
ERR_MIN_THICK = 1
ERR_MIN_RHO = 0.2


def print_model(model0, model1):
    print("                   Initial \t            Step")
    for p in model0.keys():
        if p in model1:
            print(
                "%15s %7.3g +- %-7.2g \t %7.3g +- %-7.2g"
                % (
                    p,
                    model0[p]["best"],
                    model0[p]["std"],
                    model1[p]["best"],
                    model1[p]["std"],
                )
            )
        else:
            print("%15s %7.3g +- %-7.2g" % (p, model0[p]["best"], model0[p]["std"]))


def sample_from_json_file(
    model_expt_json_file, model_err_json_file=None, prior_scale=1, set_ranges=False
):
    """
    Return the sample object described by the provided json data.

    If model_err_json is provided, it will be used to set the width of
    the prior distribution.
    """
    with open(model_expt_json_file, "r") as fd:
        expt = json.load(fd)

    err = None
    if model_err_json_file:
        with open(model_err_json_file, "r") as fd:
            err = json.load(fd)

    return sample_from_json(
        expt, model_err_json=err, prior_scale=prior_scale, set_ranges=set_ranges
    )


def sample_from_json(
    model_expt_json, model_err_json=None, prior_scale=1, set_ranges=False
):
    """
    Return the sample object described by the provided json data.

    If model_err_json is provided, it will be used to set the width of
    the prior distribution.
    """
    sample = None
    for layer in model_expt_json["sample"]["layers"]:
        # dict_keys(['type', 'name', 'thickness', 'interface', 'material', 'magnetism'])

        rho = layer["material"]["rho"]["value"]
        rho_fixed = layer["material"]["rho"]["fixed"]
        rho_limits = layer["material"]["rho"]["bounds"]["limits"]
        rho_std = 0

        irho = layer["material"]["irho"]["value"]
        irho_fixed = layer["material"]["irho"]["fixed"]
        irho_limits = layer["material"]["irho"]["bounds"]["limits"]
        irho_std = 0

        thickness = layer["thickness"]["value"]
        thickness_fixed = layer["thickness"]["fixed"]
        thickness_limits = layer["thickness"]["bounds"]["limits"]
        thickness_std = 0

        interface = layer["interface"]["value"]
        interface_fixed = layer["interface"]["fixed"]
        interface_limits = layer["interface"]["bounds"]["limits"]
        interface_std = 0

        if model_err_json:
            if layer["material"]["rho"]["name"] in model_err_json:
                if prior_scale > 0:
                    rho_std = (
                        prior_scale
                        * model_err_json[layer["material"]["rho"]["name"]]["std"]
                        + ERR_MIN_RHO
                    )
                else:
                    rho_std = 0
            if layer["material"]["irho"]["name"] in model_err_json:
                if prior_scale > 0:
                    irho_std = (
                        prior_scale
                        * model_err_json[layer["material"]["irho"]["name"]]["std"]
                        + ERR_MIN_RHO
                    )
                else:
                    irho_std = 0
            if layer["thickness"]["name"] in model_err_json:
                if prior_scale > 0:
                    thickness_std = (
                        prior_scale * model_err_json[layer["thickness"]["name"]]["std"]
                        + ERR_MIN_THICK
                    )
                else:
                    thickness_std = 0
            if layer["interface"]["name"] in model_err_json:
                if prior_scale > 0:
                    interface_std = (
                        prior_scale * model_err_json[layer["interface"]["name"]]["std"]
                        + ERR_MIN_ROUGH
                    )
                else:
                    interface_std = 0

        material = SLD(name=layer["name"], rho=rho, irho=irho)

        slab = Slab(material=material, thickness=thickness, interface=interface)

        # Set the range for each tunable parameter
        if not rho_fixed:
            if rho_std > 0:
                slab.material.rho.dev(rho_std, limits=(rho_limits[0], rho_limits[1]))
            else:
                slab.material.rho.range(rho_limits[0], rho_limits[1])
            slab.material.rho.fixed = not set_ranges
        if not irho_fixed:
            if irho_std > 0:
                slab.material.irho.dev(
                    irho_std, limits=(irho_limits[0], irho_limits[1])
                )
            else:
                slab.material.irho.range(irho_limits[0], irho_limits[1])
            slab.material.irho.fixed = not set_ranges
        if not thickness_fixed:
            print("Setting thickness")
            if thickness_std > 0:
                print(thickness_std)
                slab.thickness.dev(
                    thickness_std, limits=(thickness_limits[0], thickness_limits[1])
                )
                print(slab.thickness.distribution.std)
            else:
                slab.thickness.range(thickness_limits[0], thickness_limits[1])
            slab.thickness.fixed = not set_ranges
        if not interface_fixed:
            if interface_std > 0:
                slab.interface.dev(
                    interface_std, limits=(interface_limits[0], interface_limits[1])
                )
            else:
                slab.interface.range(interface_limits[0], interface_limits[1])
            slab.interface.fixed = not set_ranges

        sample = slab if sample is None else sample | slab
    return sample


def fix_all_parameters(expt, verbose=False):
    """
    Fix all the parameters within an Experiment object

    Parameters
    ----------
    expt : Experiment
        Experiment object to process
    verbose : bool
        If True, print out parameters that were not fixed
    """
    pars = expt.parameters()

    def _fix_parameters(item):
        if type(item) is Parameter:
            if verbose and not item.fixed:
                print("Found %s" % item)
            item.fixed = True
        elif type(item) is list:
            for p in item:
                _fix_parameters(p)
        elif type(item) is dict:
            for p, v in item.items():
                _fix_parameters(v)
        else:
            print("Found unknown parameter: %s" % item)

    _fix_parameters(pars)


def expt_from_json_file(
    model_expt_json_file: str,
    probe: "QProbe | None" = None,
    model_err_json_file: str = None,
    prior_scale: float = 1,
    set_ranges: bool = False,
    keep_original_ranges: bool = False,
):
    """
    Load an Experiment from an experiment json file.

    When iterating over data slices, the experiment will be used for data other
    that what was originally loaded to run the fit that created the json file.
    To allow for this usage, we may create a new experiment with a given probe.

    Given that we may also want to change the fit parameters, we will need the
    ability to switch off all the existing limits.

    Parameters
    ----------
    model_expt_json_file : str
        -expt.json file
    probe : QProbe
        Optional Probe object to replace the one found in the serialized Experiment
    model_err_json_file : str
        -err.json file containing the uncertainties from the previous fit
    prior_scale : float
        Optional parameter to multiply the width of the Bayesian prior by
    set_ranges : bool
        If False, all the parameters should be fixed
    keep_original_ranges : bool
        If True, the parameter ranges found in the Experiment file will be kept

    Returns
    -------
        Experiment
    """
    with open(model_expt_json_file, "rt") as input_file:
        serialized = input_file.read()
        serialized_dict = json.loads(serialized)
        expt = serialize.deserialize(serialized_dict, migration=True)

    if not keep_original_ranges:
        # Since this Experiment was created by a fit to an initial/final state,
        # it may not have the correct fit parameters. Fix all the parameters
        # and set the correct fit parameters according to the provided uncertainty file.
        fix_all_parameters(expt)

        # set_ranges and providing the err.json file are redundent information...
        # TODO refactor this
        if not set_ranges:
            pass

    # If the probe was provided, create a new experiment with it.
    if probe is not None:
        expt = Experiment(probe=probe, sample=expt.sample)

    return expt


def calculate_reflectivity(model_expt_json_file, q, q_resolution=0.025):
    """
    Reflectivity calculation using refl1d
    """
    expt = expt_from_json_file(model_expt_json_file, q, q_resolution=q_resolution)
    _, r = expt.reflectivity()
    return r
