import sys
sys.path.append('..')
import unittest
from hypothesis import given, settings, example, unlimited
import hypothesis.strategies as st
import numpy as np
from SignalProcessing import SignalAnalyserBase
from SignalProcessing import SimplePulseAnalyser
from SignalProcessing import LandauAnalyser
import pylandau

do_fast_tests = True
def conditional_decorator(input_decorator, condition):
    def output_decorator(func):
        if not condition:
            return func
        return input_decorator(func)
    return output_decorator

numsStrategy = st.one_of(
    st.integers(),
    st.floats(max_value=1e150, min_value=-1e150),
    )

class SignalAnalyserBaseTest(unittest.TestCase):
    def test_signalanalyserbase_is_an_abstractclass(self):
        with self.assertRaises(TypeError):
            a = SignalAnalyserBase(t=1.0, sig=1.0)

    def test_quality_is_an_abstractmethod(self):
        class FailingTestClass(SignalAnalyserBase):
            '''Should fail since self.quality is an abstractmethod'''
            pass

        class SucceedingTestClass(SignalAnalyserBase):
            '''Should succeed cos abstractmethod self.quality is overwritten'''
            def quality(self):
                pass

        with self.assertRaises(TypeError):
            a = FailingTestClass(t=1.0, sig=1.0)

        a = SucceedingTestClass(t=1.0, sig=1.0)

class SimplePulseAnalyserTest(unittest.TestCase):
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    @given(
        st.integers(min_value=2),
        numsStrategy.filter(lambda x: not x==0),
    )
    def test_findPeak(self, size, time_adjust):
        sig = np.concatenate((
            np.array(range(size)),
            np.array(range(size, 0, -1))
        ))
        t = np.arange(len(sig)) / time_adjust

        a = SimplePulseAnalyser(t, sig)
        peak = (size, size)
        self.assertEqual(a.peak, peak)

    @given(st.integers(min_value=4))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_halfHeight(self, size):
        sig = np.concatenate((
            np.array(range(size)),
            np.array(range(size, 0, -1))
        ))
        t = np.arange(len(sig))
        a = SimplePulseAnalyser(t, sig)
        duration = size
        self.assertEqual(a.width, duration)

        sig = np.concatenate((
            np.zeros(size),
            np.ones(size),
            np.zeros(size),
        ))
        t = np.arange(len(sig))
        a = SimplePulseAnalyser(t, sig)
        duration = size
        self.assertEqual(a.width, duration)

    @given(st.integers(min_value=4))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_calcArea(self, size):
        sig = np.concatenate((
            np.zeros(size),
            np.ones(size),
            np.zeros(size),
        ))
        t = np.arange(len(sig))
        a = SimplePulseAnalyser(t, sig)
        self.assertEqual(a.calcArea(), size)

        sig = np.concatenate((
            np.array(range(size)),
            np.array(range(size, 0, -1))
        ))
        t = np.arange(len(sig))
        a = SimplePulseAnalyser(t, sig)
        # sum of ints from 1 to n is n(n+1)/2. Do the maths.
        expected_sum = size**2
        self.assertEqual(a.calcArea(), expected_sum)

class LandauAnalyserTest(unittest.TestCase):
    @given(mpv=st.floats(min_value=100, max_value=900, allow_nan=False))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_recover_mpv(self, mpv):
        time = np.arange(0, 1e3, 0.001)
        sig = pylandau.landau(time, mpv=mpv, eta=20, A=2)

        a = LandauAnalyser(
            time,
            sig,
            p0=[mpv, 20, 2],
            bounds=([-1000, 1, 0], [1500, 600, 1e7]),
            )
        self.assertEqual(a.p_opt[0][0], mpv)
        self.assertEqual(a.p_opt[0][1], 20)
        self.assertEqual(a.p_opt[0][2], 2)

    @given(eta=st.floats(min_value=2, max_value=100, allow_nan=False))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_recover_eta(self, eta):
        time = np.arange(0, 1e3, 0.001)
        sig = pylandau.landau(time, mpv=500, eta=eta, A=2)

        a = LandauAnalyser(
            time,
            sig,
            p0=[500, eta, 2],
            bounds=([-1000, 1, 0], [1500, 600, 1e7]),
            )
        self.assertEqual(a.p_opt[0][0], 500)
        self.assertEqual(a.p_opt[0][1], eta)
        self.assertEqual(a.p_opt[0][2], 2)

    @given(A=st.floats(min_value=1e-3, max_value=100, allow_nan=False))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_recover_A(self, A):
        time = np.arange(0, 1e3, 0.001)
        sig = pylandau.landau(time, mpv=500, eta=20, A=A)

        a = LandauAnalyser(
            time,
            sig,
            p0=[500, 20, A],
            bounds=([-1000, 1, 0], [1500, 600, 1e7]),
            )
        self.assertEqual(a.p_opt[0][0], 500)
        self.assertEqual(a.p_opt[0][1], 20)
        self.assertEqual(a.p_opt[0][2], A)

    @given(mpv=st.floats(min_value=100, max_value=900, allow_nan=False))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_QoverA(self, mpv):
        time = np.arange(0, 1e3, 0.001)
        sig = pylandau.landau(time, mpv=mpv, eta=20, A=2)

        a = LandauAnalyser(
            time,
            sig,
            p0=[mpv, 20, 2],
            bounds=([-1000, 1, 0], [1500, 600, 1e7]),
            )
        Q = sum(sig) * (time[1] - time[0])
        A = 2

        self.assertEqual(a.QoverA(), Q/A)

    @given(mpv=st.floats(min_value=100, max_value=900, allow_nan=False))
    @conditional_decorator(settings(max_examples=20), do_fast_tests)
    def test_quality(self, mpv):
        time = np.arange(0, 1e3, 0.001)
        sig = pylandau.landau(time, mpv=mpv, eta=20, A=2)

        a = LandauAnalyser(
            time,
            sig,
            p0=[mpv, 20, 2],
            bounds=([-1000, 1, 0], [1500, 600, 1e7]),
            )
        Q = sum(sig) * (time[1] - time[0])
        A = 2

        self.assertEqual(a.quality(), Q/A)
