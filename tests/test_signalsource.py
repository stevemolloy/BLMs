import sys
sys.path.append('..')
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
import scipy
from SignalProcessing import SignalSource

max_scipy_examples = 10

class SignalSourceCleanTest(unittest.TestCase):
    numsStrategy = st.one_of(
        st.integers(),
        st.floats(max_value=1e150, min_value=-1e150),
    )

    def setUp(self):
        self.SS = SignalSource()

    @given(numsStrategy, numsStrategy)
    def test_cleansignal_callable(self, amp, arrival):
        a = self.SS.cleanSignal(amplitude=amp, arrivalTime=arrival)
        self.assertTrue(callable(a))

    @given(numsStrategy)
    def test_cleansignal_func_returns_float(self, t):
        a = self.SS.cleanSignal(amplitude=10, arrivalTime=200)
        self.assertIsInstance(a(t), np.ndarray)

    @given(st.lists(elements=numsStrategy, min_size=1))
    def test_cleansignal_func_return_length(self, t):
        a = self.SS.cleanSignal(amplitude=10, arrivalTime=200)
        self.assertEqual(len(t), len(a(t)))

    @given(st.floats(min_value=-1e9, max_value=1e9))
    @settings(max_examples=max_scipy_examples)
    def test_cleansignal_peak_location(self, peakLoc):
        a = self.SS.cleanSignal(amplitude=10, arrivalTime=peakLoc)
        max_x = scipy.optimize.fmin(lambda x: -a(x), peakLoc, disp=False)
        self.assertAlmostEqual(max_x[0], peakLoc)

class SignalSourceSparkTest(unittest.TestCase):
    numsStrategy = st.one_of(
        st.integers(),
        st.floats(max_value=1e150, min_value=-1e150),
    )

    def setUp(self):
        self.SS = SignalSource()

    @given(numsStrategy, numsStrategy)
    def test_spark_callable(self, amp, arrival):
        a = self.SS.spark(amplitude=10, arrivalTime=200)
        self.assertTrue(callable(a))

    @given(numsStrategy)
    def test_spark_func_returns_float(self, t):
        a = self.SS.spark(amplitude=10, arrivalTime=200)
        self.assertIsInstance(a(t), np.ndarray)

    @given(st.lists(elements=numsStrategy, min_size=1))
    def test_spark_func_return_length(self, t):
        a = self.SS.spark(amplitude=10, arrivalTime=200)
        self.assertEqual(len(t), len(a(t)))

    @given(st.floats(min_value=0, max_value=1e9))
    @settings(max_examples=max_scipy_examples)
    def test_spark_peak_location(self, peakLoc):
        a = self.SS.spark(amplitude=10, arrivalTime=peakLoc)
        max_x = scipy.optimize.fmin(lambda x: -a(x), peakLoc, disp=False)
        self.assertAlmostEqual(max_x[0], peakLoc)

class SignalSourceGlitchTest(unittest.TestCase):
    numsStrategy = st.one_of(
        st.integers(),
        st.floats(max_value=1e150, min_value=-1e150),
    )

    def setUp(self):
        self.SS = SignalSource()

    @given(numsStrategy, numsStrategy)
    def test_glitch_callable(self, amp, arrival):
        a = self.SS.glitch(amplitude=amp, arrivalTime=arrival)
        self.assertTrue(callable(a))

    @given(numsStrategy)
    def test_glitch_func_returns_float(self, t):
        a = self.SS.glitch(amplitude=10, arrivalTime=200)
        self.assertIsInstance(a(t), np.ndarray)

    @given(st.lists(elements=numsStrategy, min_size=1))
    def test_glitch_func_return_length(self, t):
        a = self.SS.glitch(amplitude=10, arrivalTime=200)
        self.assertEqual(len(t), len(a(t)))

    @given(st.floats(min_value=0, max_value=1e9))
    @settings(max_examples=max_scipy_examples)
    def test_glitch_peak_location(self, peakLoc):
        a = self.SS.glitch(amplitude=10, arrivalTime=peakLoc)
        max_x = scipy.optimize.fmin(lambda x: -a(x), peakLoc, disp=False)
        self.assertAlmostEqual(max_x[0], peakLoc)

if __name__ == '__main__':
    unittest.main()
