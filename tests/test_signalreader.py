import sys
sys.path.append('..')
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
import scipy
from functools import partial
from SignalProcessing import SignalReader

def func_that_rets_constant(t, constVal):
    try:
        return np.array([constVal for i in t])
    except TypeError:
        return constVal

class SignalReaderTest(unittest.TestCase):
    numsStrategy = st.one_of(
        st.integers(),
        st.floats(max_value=1e150, min_value=-1e150),
    )
    float_noNAN_noInf = st.floats(allow_nan=False, allow_infinity=False)
    min_len_list = st.lists(elements=numsStrategy, min_size=2)
    zero_to_sixteen = st.integers(min_value=0, max_value=16)

    @given(float_noNAN_noInf, st.integers(min_value=0))
    def test_signalreader_initialisation(self, maxVolts, numbits):
        a = SignalReader(maxVolts=maxVolts, numbits=numbits)
        self.assertAlmostEqual(a.noise, maxVolts * 2**(-numbits))

    @given(zero_to_sixteen, min_len_list)
    def test_nullsignal_nonoise_reading(self, numbits, time):
        a = SignalReader(noise=0, maxVolts=1.0, numbits=numbits)

        func = partial(func_that_rets_constant, constVal=0.0)
        r = list(a.acquireData(time, func))
        self.assertEqual(np.mean(r), 0)

    @given(zero_to_sixteen, min_len_list)
    def test_unitysignal_nonoise_reading(self, numbits, time):
        a = SignalReader(noise=0, maxVolts=10.0, numbits=numbits)

        func = partial(func_that_rets_constant, constVal=1.0)
        r = list(a.acquireData(time, func))
        self.assertAlmostEqual(np.mean(r), 1)

    @given(min_len_list)
    def test_clips_at_maxvolts(self, time):
        a = SignalReader(noise=0, maxVolts=1.0, numbits=12)

        func = partial(func_that_rets_constant, constVal=10.0)
        r = list(a.acquireData(time, func))
        self.assertAlmostEqual(max(r), 1.0)

    @given(min_len_list)
    def test_clips_at_zero(self, time):
        a = SignalReader(noise=0, maxVolts=1.0, numbits=12)

        func = partial(func_that_rets_constant, constVal=-10.0)
        r = list(a.acquireData(time, func))
        self.assertAlmostEqual(min(r), 0)

    def test_reproduces_original(self):
        num = 10000
        time = np.linspace(0, 100*np.pi, num=num)

        a = SignalReader(maxVolts=1000.0)
        r = np.array(list(a.acquireData(time, lambda t: 200*np.sin(t) + 500.0)))
        fixed = r - (200*np.sin(time) + 500)

        self.assertAlmostEqual(np.mean(fixed), 0, delta=5 * np.std(fixed)/(num**0.5))
        self.assertAlmostEqual(np.std(fixed), a.noise, places=2)

if __name__ == '__main__':
    unittest.main()
