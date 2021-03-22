import sys
sys.path.append('../scripts')
import numpy as np

import unittest
import ekf
from states import States

class TestPropagation(unittest.TestCase):
    def test_no_inputs(self):
        p0 = [0.0,0.0,0.0]
        q0 = [0.0,0.0,0.0,1.0]
        v0 = [0.0,0.0,0.0]
        ba0 = [0.0,0.0,0.0]
        bg0 = [0.0,0.0,0.0]
        cov0 = [1.0,1.0,1.0]
        xHat = States(p0,q0,v0,ba0,bg0,cov0,cov0,cov0,cov0,cov0)
        accel = [0.0,0.0,0.0]
        omega = [0.0,0.0,0.0]
        dt = 0.1
        pExpected = [0.0,0.0,0.0]
        qExpected = [0.0,0.0,0.0,1.0]
        vExpected = [0.0,0.0,0.981]
        baExpected = [0.0,0.0,0.0]
        bgExpected = [0.0,0.0,0.0]
        covExpected = [1.0,1.0,1.0]
        expectedXHat = States(pExpected,qExpected,vExpected,baExpected,bgExpected,covExpected,covExpected,covExpected,covExpected,covExpected)
        expectedG = [[0.0,0.0,0.0],
                     [0.0,0.0,0.0],
                     [0.0,0.0,0.0]]
        [xHat, G] = ekf.dynamics(xHat,accel,omega,dt)
        self.assert_state_equal(xHat,expectedXHat)
        for i in range(3):
            for j in range(3):
                self.assertEqual(G[i][j],expectedG[i][j])

    def test_111_111_inputs(self):
        p0 = [0.0,0.0,0.0]
        q0 = [0.0,0.0,0.0,1.0]
        v0 = [0.0,0.0,0.0]
        ba0 = [0.0,0.0,0.0]
        bg0 = [0.0,0.0,0.0]
        cov0 = [1.0,1.0,1.0]
        xHat = States(p0,q0,v0,ba0,bg0,cov0,cov0,cov0,cov0,cov0)
        accel = [1.0,1.0,1.0]
        omega = [1.0,1.0,1.0]
        dt = 0.1
        pExpected = [0.0,0.0,0.0]
        qExpected = [0.0523491,0.0473595,0.0523491,0.9961306]
        vExpected = [0.1,0.1,1.081]
        baExpected = [0.0,0.0,0.0]
        bgExpected = [0.0,0.0,0.0]
        covExpected = [1.0,1.0,1.0]
        expectedXHat = States(pExpected,qExpected,vExpected,baExpected,bgExpected,covExpected,covExpected,covExpected,covExpected,covExpected)
        expectedG = [[0.0,0.0,0.0],
                     [0.0,0.0,0.0],
                     [0.0,0.0,0.0]]
        [xHat, G] = ekf.dynamics(xHat,accel,omega,dt)
        self.assert_state_equal(xHat,expectedXHat)
        for i in range(3):
            for j in range(3):
                self.assertEqual(G[i][j],expectedG[i][j])

    def assert_state_equal(self,xHat,expectedXHat):
        for i in range(3):
            self.assertAlmostEqual(xHat.p[i],expectedXHat.p[i])
            self.assertAlmostEqual(xHat.v[i],expectedXHat.v[i])
            self.assertAlmostEqual(xHat.ba[i],expectedXHat.ba[i])
            self.assertAlmostEqual(xHat.bg[i],expectedXHat.bg[i])
        for j in range(4):
            self.assertAlmostEqual(xHat.q[j],expectedXHat.q[j])


if __name__ == '__main__':
    unittest.main()