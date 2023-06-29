import numpy as np

from bumps.parameter import Parameter

from refl1d.model import Slab
from refl1d.material import SLD
from refl1d.experiment import Experiment

class BayesSLD(SLD):
    def __init__(self, name="SLD", rho=0, irho=0, rho_width=0, irho_width=0):
        self.name = name
        self.rho = Parameter.default(rho, name=name+" rho",
                                     center=rho, width=rho_width)
        self.irho = Parameter.default(irho, name=name+" irho",
                                      center=irho, width=irho_width)


class BayesSlab(Slab):
    def __init__(self, material=None, thickness=0, interface=0, name=None,
                 magnetism=None, thickness_width=0, interface_width=0):
        if name is None:
            name = material.name
        self.name = name
        self.material = material
        self.thickness = Parameter.default(thickness, limits=(0, np.inf),
                                           name=name+" thickness",
                                           center=thickness,
                                           width=thickness_width)
        self.interface = Parameter.default(interface, limits=(0, np.inf),
                                           name=name+" interface",
                                           center=interface,
                                           width=interface_width)
        self.magnetism = magnetism


class BayesExperiment(Experiment):
    def _residuals(self):
        if 'residuals' not in self._cache:
            # Trigger reflectivity calculation even if there is no data to
            # compare against so that we can profile simulation code, and
            # so that simulation smoke tests are run more thoroughly.
            QR = self.reflectivity()
            if ((self.probe.polarized
                 and all(x is None or x.R is None for x in self.probe.xs))
                    or (not self.probe.polarized and self.probe.R is None)):
                resid = np.zeros(0)
            else:
                if self.probe.polarized:
                    resid = np.hstack([(xs.R - QRi[1])/xs.dR
                                       for xs, QRi in zip(self.probe.xs, QR)
                                       if xs is not None])
                else:
                    resid = (self.probe.R - QR[1])/self.probe.dR

            # Multiply by Bayesian prior
            # Look through parameters and extract probability distribution.
            # Note that we will be using log likelihood, so we are simply
            # adding the likelihoods.
            def _process(p):
                if not p.fixed and p.width>0:
                    return (p.value - p.center) / p.width
                return 0

            prior = []
            for layer in self.sample._layers:
                prior.append(_process(layer.material.rho))
                prior.append(_process(layer.material.irho))
                prior.append(_process(layer.thickness))
                prior.append(_process(layer.interface))

            resid = np.append(resid, np.asarray(prior))
            self._cache['residuals'] = resid

        return self._cache['residuals']

    def prior(self):
        if '__prior' not in self._cache:
            # Multiply by Bayesian prior
            # Look through parameters and extract probability distribution.
            # Note that we will be using log likelihood, so we are simply
            # adding the likelihoods.
            def _process(p):
                if not p.fixed and p.width>0:
                    return (p.value - p.center) / p.width
                return 0

            prior = []
            for layer in self.sample._layers:
                prior.append(_process(layer.material.rho))
                prior.append(_process(layer.material.irho))
                prior.append(_process(layer.thickness))
                prior.append(_process(layer.interface))

            self._cache['__prior'] = 0.5*np.sum(np.asarray(prior)**2)

        return self._cache['__prior']

    def nllf(self):
            """
            Return the -log(P(data|model)).
            Using the assumption that data uncertainty is uncorrelated, with
            measurements normally distributed with mean R and variance dR**2,
            this is just sum( resid**2/2 + log(2*pi*dR**2)/2 ).
            The current version drops the constant term, sum(log(2*pi*dR**2)/2).
            """
            #if 'nllf_scale' not in self._cache:
            #    if self.probe.dR is None:
            #        raise ValueError("No data from which to calculate nllf")
            #    self._cache['nllf_scale'] = np.sum(np.log(2*pi*self.probe.dR**2))
            # TODO: add sigma^2 effects back into nllf; only needs to be calculated
            # when dR changes, so maybe it belongs in probe.
            
            return 0.5*np.sum(self.residuals()**2) + self.prior() # + self._cache['nllf_scale']

    def save(self, basename):
        self.save_profile(basename)
        self.save_refl(basename)
        self.save_json(basename)
        
        json_file = basename + "-bayes.dat"
        with open(json_file, 'w') as fid:
            _resid = self.residuals()
            nll = 0.5*np.sum(_resid**2)
            points = len(_resid)
            fid.write("NLL: %g\n" % nll)
            fid.write("NLPrior: %g\n" % self.prior())
            fid.write("Points: %g\n" % points)
            fid.write("Chi2: %g\n" % (2*nll/points))
