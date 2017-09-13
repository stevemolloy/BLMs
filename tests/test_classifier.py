import sys
sys.path.append('..')
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
from SignalProcessing import Classifier

class ClassifierTest(unittest.TestCase):
    @given(st.integers(min_value=2))
    def test_perfect_classifier(self, num):
        events = np.ones(num)
        nonevents = np.concatenate((np.zeros(num), 2*np.ones(num)))

        a = Classifier(events=events, nonevents=nonevents, cuts=(0.5, 1.5))

        self.assertEqual(a.TP/a.P, 1.0)
        self.assertEqual(a.TN/a.N, 1.0)
        self.assertAlmostEqual(a.MCC, 1.0)

    @given(st.integers(min_value=2))
    def test_perfectlywrong_classifier(self, num):
        nonevents = np.ones(num)
        events = np.concatenate((np.zeros(num), 2*np.ones(num)))

        a = Classifier(events=events, nonevents=nonevents, cuts=(0.5, 1.5))

        self.assertEqual(a.TP/a.P, 0.0)
        self.assertEqual(a.TN/a.N, 0.0)
        self.assertAlmostEqual(a.MCC, -1.0)

    @given(st.integers(min_value=2))
    def test_useless_classifier(self, num):
        events = np.concatenate((np.zeros(num), np.ones(num)))
        nonevents = np.concatenate((np.zeros(num), np.ones(num)))

        a = Classifier(events=events, nonevents=nonevents, cuts=(0.5, 1.5))

        self.assertEqual(a.TP/a.P, 0.5)
        self.assertEqual(a.TN/a.N, 0.5)
        self.assertAlmostEqual(a.MCC, 0.0)

    @given(st.integers(min_value=2))
    def test_false_positives(self, num):
        events = np.concatenate((np.ones(num), np.ones(num)))
        nonevents = np.concatenate((np.zeros(num), np.ones(num)))
        a = Classifier(events=events, nonevents=nonevents, cuts=(0.5, 1.5))
        self.assertEqual(a.FP, num)
        self.assertEqual(a.FN, 0)

        events = np.concatenate((np.ones(num), np.ones(num)))
        nonevents = np.concatenate((1.75*np.ones(num), np.ones(num)))
        a = Classifier(events=events, nonevents=nonevents, cuts=(0.5, 1.5))
        self.assertEqual(a.FP, num)
        self.assertEqual(a.FN, 0)
