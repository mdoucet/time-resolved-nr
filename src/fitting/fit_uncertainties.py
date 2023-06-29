"""
  Show an uncertainty band for an SLD profile.
  This currently works for inverted geometry and fixed substrate roughness, as it aligns
  the profiles to that point before doing the statistics.
"""
import os
import time
import refl1d
from refl1d.names import *
from bumps import dream

import numpy as np

class Accumulator(object):
    def __init__(self, name='', z_min=-10, z_max=450, z_step=2.0, is_magnetic=True):
        self.z = np.arange(z_min, z_max, z_step)
        self.z_mid = [(self.z[i+1]+self.z[i])/2.0 for i in range(len(self.z)-1)]
        self.summed = np.zeros(len(self.z)-1)
        self.sq_summed = np.zeros(len(self.z)-1)
        self.m_summed = np.zeros(len(self.z)-1)
        self.m_sq_summed = np.zeros(len(self.z)-1)
        self.counts = np.zeros(len(self.z)-1)
        self.z_step = z_step
        self.name = name
        self.is_magnetic = is_magnetic
        
        # Accumulate in an array
        self.sld_models = []

    def add(self, z, rho, rhoM):
        """ Add a model to the average """
        if rhoM is None:
            self.is_magnetic = False
        
        # Compute the average step size so we can normalize after rebinning
        average_step = np.fabs(np.asarray([z[i+1]-z[i] for i in range(0, len(z)-2, 2) if z[i]>0])).mean()
        
        z_ = np.asarray(z)
        rho_ = np.asarray(rho[:-1])
        r_out = refl1d.rebin.rebin(z_, rho_, self.z)

        r_out = r_out * average_step / self.z_step

        # Ensure that we continue the SLD profile in outgoing medium
        for i in range(len(r_out)):
            if self.z[i+1] >= z[0]:
                r_out[i] = rho[0]
        
        self.summed += r_out
        self.sq_summed += r_out * r_out

        if self.is_magnetic:
            rhoM_ = np.asarray(rhoM[:-1])
            rM_out = refl1d.rebin.rebin(z_, rhoM_, self.z)
            rM_out = rM_out * average_step / self.z_step
            self.m_summed += rM_out
            self.m_sq_summed += rM_out * rM_out

        _counts = np.ones(len(self.z[:-1]))
        #print(z[0])
        self.counts += _counts

        self.sld_models.append(r_out)

    def quantiles(self, cl=0.90):
        from scipy.stats.mstats import mquantiles
        prob = np.asarray([100.0 - cl, 100.0 + cl])/200.0
        _q = mquantiles(self.sld_models, prob=[prob], axis=0)
        return self.z_mid, _q
        
    def mean(self):
        _counts = np.asarray([ 1 if c==0 else c for c in self.counts ])
        avg = self.summed / _counts
        sq_avg = self.sq_summed / _counts
        sig = np.sqrt(np.fabs(np.fabs(sq_avg) - avg*avg))

        return self.z_mid, avg, sig

    def mean_magnetism(self):
        if not self.is_magnetic:
            return self.z_mid, np.zeros(len(self.z_mid)), np.zeros(len(self.z_mid))

        _counts = np.asarray([ 1 if c==0 else c for c in self.counts ])
        avg = self.m_summed / _counts
        sq_avg = self.m_sq_summed / _counts
        sig = np.sqrt(sq_avg - avg*avg)

        return self.z_mid, avg, sig

def load_bumps(file_path, problem, trim=1000, state=None, z_min=0, z_max=450.0):
    """
        Use bumps to load MC
    """
    model_list = [problem]
    if hasattr(problem, '_models'):
        model_list = problem._models

    acc = [Accumulator(m.fitness.name,z_min=z_min, z_max=z_max) for m in model_list]
    
    t0 = time.time()
    if state is None:
        state = dream.state.load_state(file_path)
    state.mark_outliers()

    # If we have a population, only pick 1000 of them
    if state.draws>trim:
        print("Too many points: pruning down")
        portion = trim / state.draws
        drawn = state.draw(portion=portion)
    else:
        drawn = state.draw(portion=1.0)

    print("MC file read: %s sec" % (time.time()-t0))
    pts = np.asarray(drawn.points)

    for i, p in enumerate(pts):
        problem.setp(p)
        # Loop over models (Experiments)
        for j, model in enumerate(model_list):
            if model.fitness.ismagnetic:
                z, r, _, rM, _ = model.fitness.magnetic_smooth_profile()
                acc[j].add(z[-1]-z, r, rM)
            else:
                z, r, _ = model.fitness.smooth_profile()
                acc[j].add(z[-1]-z, r, None)

    print("Done %s sec" % (time.time()-t0))
    return acc
