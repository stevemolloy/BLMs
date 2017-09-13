
# coding: utf-8

# # Neutron Beam Loss Monitor processing
# ## Introduction
# The nBLM's being produced by CEA need to have processing algorithms defined -- both at the level of individual monitors, but also taking advantage of multiple monitors to better understand losses throughout the machine.
# 
# 

# In[1]:


import pylandau
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import LogColorMapper, LogTicker, ColorBar, ColorMapper, LinearColorMapper
import numpy as np
import random
from scipy.stats import norm
from functools import partial
from pandas import cut
from scipy.optimize import curve_fit
from itertools import chain
import ipywidgets as widgets
from IPython.display import display
import abc # Abstract Base Class

class ProgressBar(widgets.IntProgress):
    def __init__(self, maxVal):
        return super().__init__(
            value=0,
            min=0,
            max=maxVal,
            step=1,
            description='',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )

output_notebook()


# ## Getting started
# First we create some Python classes.
# 
# The following properly simulates the expected signals (real neutron events, noise glitches above the background noise, and HV sparks) from an nBLM.  This class is used to generate fake data when real data isn't available.

# In[2]:


class SignalSource:
    '''A class whose methods return anonymous functions that represent different types of signal'''
    def cleanSignal(self, amplitude, arrivalTime):
        '''Returns a function that represents a real, neutron-based, signal'''
        return partial(pylandau.landau, mpv=arrivalTime, eta=random.gauss(20, 3), A=amplitude)
    
    def spark(self, amplitude, arrivalTime):
        '''Returns a function that represents a spark event in the HV'''
        width = random.gauss(35, 8)
        return partial(pylandau.landau, mpv=arrivalTime, eta=width, A=amplitude)
    
    def glitch(self, amplitude, arrivalTime):
        '''Returns a function that represents a noise glitch above background'''
        def pdffunc(t):
            rv = norm(loc=arrivalTime, scale=20)
            return (amplitude * 20 * (2*np.pi)**0.5) * rv.pdf(t)
        return pdffunc


# The following class takes the fake data, and simulates the action of a ADC on it.  That is, it adds features normally found in the response of digitising electronics (noise, clipping, and digitization).

# In[3]:


class SignalReader:
    """
    A class that mimics the signal obtained by a digitizer, including noise, clipping, and
    digitizastion to a fixed number of bits.
    """
    def __init__(self, noise=None, maxVolts=1.0, numbits=12):
        if not noise:
            self.noise = 2**-numbits
        else:
            self.noise = noise
        self.maxVolts = maxVolts
        self.numbits = numbits
        self.bitvalues = np.arange(start=0.0, stop=maxVolts, step=maxVolts/(2**numbits))
    
    def acquireData(self, timeArray, pulseFunction):
        '''
        This function is a generator.
        Yields the values recorded by the digitizer at a series of time steps.  The values returned
        include the noise, clipping, and digitization.
        '''
        sig = self._clipSignal(pulseFunction(timeArray))
        sig = self._clipSignal(sig + np.random.normal(0, self.noise, len(sig)), nullNegs=False)
        inds, bins = cut(sig, np.arange(start=-1e-9, stop=self.maxVolts+2**-self.numbits, step=2**-self.numbits),
                         labels=False, retbins=True)
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


# ## Making some signals
# Now that we have some Python machinery to allow us to generate signals, we can start work.
# 
# We create a 12-bit digitizer that clips at 0.35 V, and an RMS noise level of 3 mV.  Loop a significant number of times, each time generating a clean signal, a noise glitch, and a spark, each of which has randomly chosen attributes.

# In[4]:


loopIterations = 500

progress = ProgressBar(maxVal = loopIterations)
progress.layout.display = 'flex'
display(progress)

time = np.arange(start=0.0, stop=1000.0, step=0.1)
a = SignalSource()
b = SignalReader(noise=3e-3, numbits=12, maxVolts=0.35)

amp = np.random.uniform(low=0, high=0.4, size=loopIterations)
arrivalTime = np.random.uniform(low=0, high=1000, size=loopIterations)

cleansig = [[None]*len(time)]*loopIterations
nonsig = [[None]*len(time)]*loopIterations*2
for i in range(loopIterations):
    clean = a.cleanSignal(amplitude=amp[i], arrivalTime=arrivalTime[i])
    spark = a.spark(amplitude=amp[i], arrivalTime=arrivalTime[i])
    glitch = a.glitch(amplitude=amp[i]/10, arrivalTime=arrivalTime[i])
    
    cleansig[i] = [n for n in b.acquireData(time, clean)]
    nonsig[i*2] = [n for n in b.acquireData(time, spark)]
    nonsig[i*2 + 1] = [n for n in b.acquireData(time, glitch)]
    
    progress.value = i
    progress.description = '{} / {}'.format(i*3 + 1, loopIterations*3)
    
progress.value = progress.max
progress.bar_style = 'success'
progress.description = 'Done!'


# ## Analysis
# Each signal is analysed with one of the following analysis classes (looking for the real signal), and a discriminant is calculated and stored.
# 
# The following classes are used to analyse the data.  After instantiation, the ''quality'' method will return a value to be used to discriminate between the real and spurious signals.

# In[5]:


class SignalAnalyserBase(metaclass=abc.ABCMeta):
    '''
    An abstract base class for the signal analysis.
    This allows the analysis techniques to be changed by subclassing and dropping the new class
    into the analysis.
    '''
    def __init__(self, t, sig):
        self.t = t
        self.sig = sig
        self.p_opt = None
    
    @abc.abstractmethod
    def quality(self):
        pass
    
class simplePulseAnalyser(SignalAnalyserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.findPeak()
        self.halfHeight()
        self.calcArea()
        
    def findPeak(self):
        self.peak = [np.argmax(self.sig), max(self.sig)]
    
    def halfHeight(self):
        halfheight = self.peak[1]/2
        bottomEnd, topEnd = 0, len(self.sig)-1
        
        for i in range(self.peak[0], len(self.sig)):
            if self.sig[i] < halfheight:
                topEnd = i
                break
        
        for i in range(self.peak[0], 0, -1):
            if self.sig[i] < halfheight:
                bottomEnd=i
                break
        
        self.width = self.t[topEnd] - self.t[bottomEnd]
        
    def calcArea(self):
        self.area = sum(self.sig) * (self.t[1] - self.t[0])
        
    def QoverA(self):
        return self.area / self.peak[1]
    
    def quality(self):
        return self.width
    
class LandauAnalyser(SignalAnalyserBase):
    def fit(self):
        '''Fits the signal to a Landau distribution and sets the relevant coefficients'''
        def func(t, mpv, eta, A):
            return pylandau.landau(t, mpv, eta, A)
        
        self.p_opt = curve_fit(func, time, self.sig, p0=[400, 20, 0.1], bounds=([0, 1, 0], [1000, 100, 10]))
    
    def QoverA(self):
        """
        Calculates the charge (Q) normalised by the amplitude (A) under the Landau curve represented
        by the fit coefficients
        """
        self.fit()
        
        t2 = self.t[1:] + self.t[-1]
        t = np.concatenate((self.t, t2))
        fitData = pylandau.landau(self.t, self.p_opt[0][0], self.p_opt[0][1], self.p_opt[0][2])
        Q = sum(fitData) * (self.t[1]-self.t[0])
        A = self.p_opt[0][2]
        return Q/A
    
    def quality(self):
        return self.QoverA()


# Set 'analysisClass' to the class to be used for the analysis.

# In[6]:


analysisClass = simplePulseAnalyser


# Now loop around all of the signals, and store the value of the discriminant for each one.

# In[7]:


sigCount = len(cleansig)
progress = ProgressBar(maxVal = sigCount + len(nonsig))
progress.layout.display = 'flex'
display(progress)

quality_clean = np.zeros(sigCount)

for sig in cleansig:
    progress.value += 1
    progress.description = '{} / {}'.format(progress.value, progress.max)
    
    analyser = analysisClass(
        t = time,
        sig = sig
    )
    try:
        quality_clean[progress.value-1] = analyser.quality()
    except RuntimeError:
        quality_clean[progress.value-1] = np.nan
    
sigCount = len(nonsig)
progress.max = sigCount

quality_nonsig = np.zeros(sigCount)

for sig in nonsig:
    progress.value += 1
    progress.description = '{} / {}'.format(progress.value, progress.max)
    
    analyser = analysisClass(
        t = time,
        sig = sig
    )
    try:
        quality_nonsig[progress.value-1] = analyser.quality()
    except RuntimeError:
        quality_nonsig[progress.value-1] = np.nan
        
progress.value = progress.max
progress.bar_style = 'success'
progress.description = 'Done!'


# Remove any data points that were set to NaN (due to, e.q., a failing fit).

# In[8]:


quality_clean = quality_clean[~np.isnan(quality_clean)]
quality_nonsig = quality_nonsig[~np.isnan(quality_nonsig)]


# ## Results
# Now bin the data and plot the discriminant for the two populations (real events, and non-events) in stacked bars.

# In[9]:


f = figure(width=900, height=450, tools="pan,wheel_zoom,box_zoom,reset,hover")

allData = np.concatenate((quality_clean, quality_nonsig))

bins = range(0, 550, 5)
bins = np.linspace(start=0, stop=max(allData), num=int(len(allData)/25))

histc, binsc = np.histogram(quality_clean, bins=bins)
histg, binsg = np.histogram(quality_nonsig, bins=bins)

f.quad(top=histc, bottom=0, left=binsc[:-1], right=binsc[1:], color='blue')
f.quad(top=histg+histc, bottom=histc, left=binsg[:-1], right=binsg[1:], color='red')

show(f)


# ## Quality of discrimination
# To figure out how good a figure-of-merit is at discriminating good pulses from bad ones, we use the following Python class.  This performs the cuts on the set of events and the set of non-events, and provides methods to list and count the number of true-positives, false-positives, true-negatives, and false-negatives.
# 
# There is no one number that quantifies the goodness of a binary discriminant, however I have chosen to use the [Matthew's Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient "Wikipedia: MCC").  This class provides a method to return that.

# In[10]:


class Classifier:
    def __init__(self, events, nonevents, cuts):
        self.events = events
        self.nonevents = nonevents
        self.cuts = cuts
        
        self.TP = ((cuts[0] < events) & (events < cuts[1])).sum()
        self.TN = (~((cuts[0] < nonevents) & (nonevents < cuts[1]))).sum()
        self.FP = ((cuts[0] < nonevents) & (nonevents < cuts[1])).sum()
        self.FN = (~((cuts[0] < events) & (events < cuts[1]))).sum()
    
        self.P = self.TP + self.FP
        self.N = self.TN + self.FN
        
        self.MCC = self._MCC()
    
    def _MCC(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN
        
        numerator = TP*TN - FP*FN
        denom = (self.P * (TP+FN) * (TN+FP) * self.N)**0.5
        if denom==0:
            return numerator
        return numerator/denom


# Scan a range of cuts, and store the resulting MCC in order to determine the best values to use.

# In[11]:


progress = ProgressBar(maxVal = 150)
progress.layout.display = 'flex'
display(progress)

MCC = np.zeros((150, 250)) * np.nan
for cutlow in range(150):
    progress.value = cutlow
    progress.description = '{} / 150'.format(cutlow)
    for cuthigh in range(cutlow+2, 250):
        cl = Classifier(events=quality_clean, nonevents=quality_nonsig, cuts=[cutlow, cuthigh])
        MCC[cutlow, cuthigh] = cl.MCC

progress.description = 'Done!'
progress.bar_style = 'success'
color_mapper = LinearColorMapper(palette="Plasma256", low=-1.0, high=1.0)

p = figure(width=900, height=450, x_range=(0, 250), y_range=(0, 150))
p.image(image=[MCC], x=0, y=0, dw=250, dh=150, color_mapper=color_mapper)

color_bar = ColorBar(color_mapper=color_mapper, location=(0,0))

p.add_layout(color_bar, 'left')

show(p)


# Finally, based on the optimal spot in the figure above, calculate the fraction of true-positives, etc., for this dataset.

# In[12]:


cuts = np.unravel_index(np.nanargmax(MCC), MCC.shape)
cl = Classifier(events=quality_clean, nonevents=quality_nonsig, cuts=cuts)

print('cuts = ', cuts)
print(' ')
print('TP = ', 100*cl.TP/cl.P, '%')
print('TN = ', 100*cl.TN/cl.N, '%')
print('FP = ', 100*cl.FP/cl.P, '%')
print('FN = ', 100*cl.FN/cl.N, '%')
print(' ')
print('MCC = ', cl.MCC)

