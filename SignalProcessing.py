import pylandau
import numpy as np
import random
from scipy.stats import norm
from functools import partial
from pandas import cut
from scipy.optimize import curve_fit
import abc  # Abstract Base Class

class SignalSource:
    '''
    A class whose methods return anonymous functions that represent different
    types of signal
    '''
    def cleanSignal(self, amplitude, arrivalTime):
        '''
        Returns a function that represents a real, neutron-based, signal
        '''
        partial_landau = partial(
            pylandau.landau,
            mpv=arrivalTime,
            eta=random.gauss(20, 3),
            A=amplitude,
        )
        def func(t):
            t = self._fix_pylandau_input(t)
            return partial_landau(t)
        return func

    def spark(self, amplitude, arrivalTime):
        '''
        Returns a function that represents a spark event in the HV
        '''
        width = random.gauss(35, 8)
        partial_landau = partial(
            pylandau.landau,
            mpv=arrivalTime,
            eta=width,
            A=amplitude
        )
        def func(t):
            t = self._fix_pylandau_input(t)
            return partial_landau(t)
        return func

    def glitch(self, amplitude, arrivalTime):
        '''
        Returns a function that represents a noise glitch above background
        '''
        def pdffunc(t):
            t = self._fix_pylandau_input(t)
            rv = norm(loc=arrivalTime, scale=20)
            return (amplitude * 20 * (2 * np.pi)**0.5) * rv.pdf(t)
        return pdffunc

    @staticmethod
    def _fix_pylandau_input(t):
        try:
            t = np.array([float(i) for i in t])
        except TypeError:
            t = np.array(float(t))
        if not t.shape:
            t.shape = (1,)
        return t

class SignalReader:
    '''
    A class that mimics the signal obtained by a digitizer, including noise,
    clipping, and digitizastion to a fixed number of bits.
    '''

    def __init__(self, noise=None, maxVolts=1.0, numbits=12):
        if noise is None:
            self.noise = 2**-numbits * maxVolts
        else:
            self.noise = noise
        self.maxVolts = maxVolts
        self.numbits = numbits

    def acquireData(self, timeArray, pulseFunction):
        '''
        This function is a generator.
        Yields the values recorded by the digitizer at a series of time steps.
        The values returned include the noise, clipping, and digitization.
        '''
        sig = self._clipSignal(pulseFunction(timeArray))
        sig = self._clipSignal(
            sig + np.random.normal(0, self.noise, len(sig)), nullNegs=False)
        inds, bins = cut(
            sig,
            np.arange(start=-1e-9,
                stop=self.maxVolts + 2**-self.numbits,
                step=2**-self.numbits),
            labels=False,
            retbins=True
        )
        bins[0] = 0.0
        for i in bins[inds]:
            yield i

    def _clipSignal(self, sig, nullNegs=True):
        sig[sig > self.maxVolts] = self.maxVolts
        if nullNegs:
            sig[sig < 0] = 0
        else:
            sig[sig < 0] = -sig[sig < 0]
        return sig

class SignalAnalyserBase(metaclass=abc.ABCMeta):
    '''
    An abstract base class for the signal analysis.
    This allows the analysis techniques to be changed by subclassing and
    dropping the new class into the analysis.
    '''
    def __init__(self, t, sig):
        self.t = t
        self.sig = sig

    @abc.abstractmethod
    def quality(self):
        pass

class SimplePulseAnalyser(SignalAnalyserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.findPeak()
        self.halfHeight()

    def findPeak(self):
        self.peak = (np.argmax(self.sig), max(self.sig))

    def halfHeight(self):
        halfheight = self.peak[1] / 2
        bottomEnd, topEnd = 0, len(self.sig) - 1
        halfStep = (self.t[1] - self.t[0]) / 2

        for i in range(self.peak[0], len(self.sig)):
            if self.sig[i] == halfheight:
                topEnd = self.t[i]
                break
            elif self.sig[i] < halfheight:
                topEnd = self.t[i] - halfStep
                break

        for i in range(self.peak[0], 0, -1):
            if self.sig[i] == halfheight:
                bottomEnd = self.t[i]
                break
            elif self.sig[i] < halfheight:
                bottomEnd = self.t[i] + halfStep
                break
        self.width = topEnd - bottomEnd

    def calcArea(self):
        return sum(self.sig) * (self.t[1] - self.t[0])

    def QoverA(self):
        return self.calcArea() / self.peak[1]

    def quality(self):
        return self.width

class LandauAnalyser(SignalAnalyserBase):
    def __init__(self, t, sig, p0=[2.5e-8, 10, 0.1], bounds=([-1000, 1, 0], [1000, 100, 10])):
        super().__init__(t, sig)
        self.p0 = p0
        self.bounds = bounds

        self.fit()

    def fit(self):
        '''
        Fits the signal to a Landau distribution and sets the relevant
        coefficients
        '''
        def func(t, mpv, eta, A):
            return pylandau.landau(t, mpv, eta, A)

        try:
            self.p_opt = curve_fit(
                func,
                self.t * 1e9,
                self.sig,
                p0=self.p0,
                bounds=self.bounds,
                )
        except RuntimeError:
            self.p_opt = None

    def QoverA(self):
        '''
        Calculates the charge (Q) normalised by the amplitude (A) under the
        Landau curve represented by the fit coefficients
        '''
        t2 = self.t[1:] + self.t[-1]
        t = np.concatenate((self.t, t2))
        try:
            fitData = pylandau.landau(
                self.t,
                self.p_opt[0][0],
                self.p_opt[0][1],
                self.p_opt[0][2],
                )
            Q = sum(fitData) * (self.t[1] - self.t[0])
            A = self.p_opt[0][2]
            return Q / A
        except TypeError:
            return np.nan

    def quality(self):
        return self.QoverA()

class Classifier:
    def __init__(self, events, nonevents, cuts):
        self.events = events
        self.nonevents = nonevents
        self.cuts = cuts

        self.TP = 1.0 * ((cuts[0] < events) & (events < cuts[1])).sum()
        self.TN = 1.0 * (~((cuts[0] < nonevents) & (nonevents < cuts[1]))).sum()
        self.FP = 1.0 * ((cuts[0] < nonevents) & (nonevents < cuts[1])).sum()
        self.FN = 1.0 * (~((cuts[0] < events) & (events < cuts[1]))).sum()

        self.P = self.TP + self.FP
        self.N = self.TN + self.FN

        self.MCC = self._MCC()

    def _MCC(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        numerator = TP * TN - FP * FN
        denom = (self.P * (TP + FN) * (TN + FP) * self.N)**0.5
        if denom == 0:
            return numerator
        return numerator / denom
