import os
import json
import numpy as np

import refl1d
from refl1d.names import QProbe, Parameter

from .bayes_experiment import BayesSLD as SLD
from .bayes_experiment import BayesSlab as Slab
from .bayes_experiment import BayesExperiment as Experiment

ERR_MIN_ROUGH = 3
ERR_MIN_THICK = 5
ERR_MIN_RHO = 0.2

def print_model(model0, model1):
    print("                   Initial \t            Step")
    for p in model0.keys():
        if p in model1:
            print("%15s %7.3g +- %-7.2g \t %7.3g +- %-7.2g" % (p, model0[p]['best'], model0[p]['std'],
                                                               model1[p]['best'], model1[p]['std']))
        else:
            print("%15s %7.3g +- %-7.2g" % (p, model0[p]['best'], model0[p]['std']))


def sample_from_json_file(model_expt_json_file, model_err_json_file=None,
                          prior_scale=1, set_ranges=False):
    """
        Return the sample object described by the provided json data.

        If model_err_json is provided, it will be used to set the width of
        the prior distribution.
    """
    with open(model_expt_json_file, 'r') as fd:
        expt = json.load(fd)

    err = None
    if model_err_json_file:
        with open(model_err_json_file, 'r') as fd:
            err = json.load(fd)

    return sample_from_json(expt, model_err_json=err,
                            prior_scale=prior_scale, set_ranges=set_ranges)

def sample_from_json(model_expt_json, model_err_json=None, prior_scale=1, set_ranges=False):
    """
        Return the sample object described by the provided json data.

        If model_err_json is provided, it will be used to set the width of
        the prior distribution.
    """
    sample = None
    for layer in model_expt_json['sample']['layers']:
        # dict_keys(['type', 'name', 'thickness', 'interface', 'material', 'magnetism'])

        rho = layer['material']['rho']['value']
        rho_fixed = layer['material']['rho']['fixed']
        rho_limits = layer['material']['rho']['bounds']['limits']
        rho_std = 0

        irho = layer['material']['irho']['value']
        irho_fixed = layer['material']['irho']['fixed']
        irho_limits = layer['material']['irho']['bounds']['limits']
        irho_std = 0

        thickness = layer['thickness']['value']
        thickness_fixed = layer['thickness']['fixed']
        thickness_limits = layer['thickness']['bounds']['limits']
        thickness_std = 0

        interface = layer['interface']['value']
        interface_fixed = layer['interface']['fixed']
        interface_limits = layer['interface']['bounds']['limits']
        interface_std = 0

        if model_err_json:
            if layer['material']['rho']['name'] in model_err_json:
                if prior_scale > 0:
                    rho_std = prior_scale*model_err_json[layer['material']['rho']['name']]['std'] + ERR_MIN_RHO
                else:
                    rho_std = 0
            if layer['material']['irho']['name'] in model_err_json:
                if prior_scale > 0:
                    irho_std = prior_scale*model_err_json[layer['material']['irho']['name']]['std'] + ERR_MIN_RHO
                else:
                    irho_std = 0
            if layer['thickness']['name'] in model_err_json:
                if prior_scale > 0:
                    thickness_std = prior_scale*model_err_json[layer['thickness']['name']]['std'] + ERR_MIN_THICK
                else:
                    thickness_std = 0
            if layer['interface']['name'] in model_err_json:
                if prior_scale > 0:
                    interface_std = prior_scale*model_err_json[layer['interface']['name']]['std'] + ERR_MIN_ROUGH
                else:
                    interface_std = 0

        material = SLD(name=layer['name'], rho=rho, irho=irho,
                       rho_width=rho_std, irho_width=irho_std)

        slab = Slab(material=material,
                    thickness=thickness, thickness_width=thickness_std,
                    interface=interface, interface_width=interface_std)

        # Set the range for each tunable parameter
        if set_ranges:
            if not rho_fixed:
                slab.material.rho.range(rho_limits[0], rho_limits[1])
            if not irho_fixed:
                slab.material.irho.range(irho_limits[0], irho_limits[1])
            if not thickness_fixed:
                slab.thickness.range(thickness_limits[0], thickness_limits[1])
            if not interface_fixed:
                slab.interface.range(interface_limits[0], interface_limits[1])

        if sample is None:
            sample = slab
        else:
            sample = sample | slab

    return sample


def expt_from_json_file(model_expt_json_file, q=None, q_resolution=0.025, probe=None,
                        model_err_json_file=None, prior_scale=1, set_ranges=False):
    """
        Return the experiment object described by the provided json data.

        If model_err_json is provided, it will be used to set the width of
        the prior distribution.
    """
    with open(model_expt_json_file, 'r') as fd:
        expt = json.load(fd)

    err = None
    if model_err_json_file:
        with open(model_err_json_file, 'r') as fd:
            err = json.load(fd)

    return expt_from_json(expt, q=q, q_resolution=q_resolution, probe=probe,
                          model_err_json=err, prior_scale=prior_scale,
                          set_ranges=set_ranges)


def expt_from_json(model_expt_json, q=None, q_resolution=0.025, probe=None,
                   model_err_json=None, prior_scale=1, set_ranges=False):
    """
        Return the experiment object described by the provided json data.

        If model_err_json is provided, it will be used to set the width of
        the prior distribution.
    """
    # The QProbe object represents the beam
    if probe is None:
        zeros = np.zeros(len(q))
        dq = q_resolution * q
        probe = QProbe(q, dq, data=None)#(zeros, zeros))

    sample = sample_from_json(model_expt_json,
                              model_err_json=model_err_json,
                              prior_scale=prior_scale, set_ranges=set_ranges)

    intensity = model_expt_json['probe']['intensity']['value']
    intensity_fixed = model_expt_json['probe']['intensity']['fixed']
    intensity_limits = model_expt_json['probe']['intensity']['bounds']['limits']
    intensity_std = 0

    background = model_expt_json['probe']['background']['value']
    background_fixed = model_expt_json['probe']['background']['fixed']
    background_limits = model_expt_json['probe']['background']['bounds']['limits']
    background_std = 0

    if model_err_json:
        if model_expt_json['probe']['intensity']['name'] in model_err_json:
            intensity_std = model_err_json[model_expt_json['probe']['intensity']['name']]['std']
        if model_expt_json['probe']['background']['name'] in model_err_json:
            background_std = model_err_json[model_expt_json['probe']['background']['name']]['std']

    probe.intensity = Parameter(value=intensity,
                                center=intensity, width=intensity_std,
                                name=model_expt_json['probe']['intensity']['name'])

    probe.background = Parameter(value=background,
                                 center=background, width=background_std,
                                 name=model_expt_json['probe']['background']['name'])
    if set_ranges:
        if not background_fixed:
            probe.intensity.range(background_limits[0], background_limits[1])
        if not intensity_fixed:
            probe.background.range(intensity_limits[0], intensity_limits[1])

    return Experiment(probe=probe, sample=sample)


def calculate_reflectivity(model_expt_json_file, q, q_resolution=0.025):
    """
        Reflectivity calculation using refl1d
    """
    expt = expt_from_json_file(model_expt_json_file, q, q_resolution=0.025)
    _, r = expt.reflectivity()
    return r

def print_parameters(model_expt_json_file, model_err_json_file=None, latex=True):

    mm = '$' if latex else ''
    sep = '&' if latex else ''
    pm = '\pm' if latex else '+-'

    # Load the fit parameter information
    if model_err_json_file is not None:
        with open(model_err_json_file) as fd:
            err_data = json.load(fd)
    else:
        err_data = {}

    # Load the full model
    with open(model_expt_json_file) as fd:
        m = json.load(fd)
        print("%-15s %-15s %-15s %-15s" % ("Name", "Thickness", "SLD", "Roughness"))
        for layer in m['sample']['layers']:
            layer_str = "%-15s" % layer['name']

            # Thickness
            par_name = layer['name']+' thickness'

            if par_name in err_data:
                layer_str += '%s%3.1f %s %3.1f%s %s' % (mm, err_data[par_name]['best'], pm, err_data[par_name]['std'], mm, sep)
            else:
                layer_str += '%s%3.1f%s           %s' % (mm, layer['thickness']['value'], mm, sep)

            # SLD
            par_name = layer['name']+' rho'
            if par_name in err_data:
                layer_str += '%s%3.2f %s %3.2f%s %s' % (mm, err_data[par_name]['best'], pm, err_data[par_name]['std'], mm, sep)
            else:
                layer_str += '%s%3.2f%s           %s' % (mm, layer['material']['rho']['value'], mm, sep)

            # Roughness
            par_name = layer['name']+' interface'
            if par_name in err_data:
                layer_str += '%s%3.2f %s %3.2f%s' % (mm, err_data[par_name]['best'], pm, err_data[par_name]['std'], mm)
            else:
                layer_str += '%s%3.2f%s' % (mm, layer['interface']['value'], mm)

            print(layer_str)
