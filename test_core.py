import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import validate, kernel, statevectors, cci, fuzzy, ROOT

class CoreTests(unittest.TestCase):
    def setUp(self):self.df=pd.read_csv(ROOT/'data/demo_features.csv')
    def test_subject_split(self):
        self.assertTrue(validate(self.df))
        self.assertEqual(self.df.groupby('split').subject_id.nunique().to_dict(),{'test':4,'train':19,'validation':4})
        bad=self.df.copy();bad.loc[0,'split']='test'
        with self.assertRaises(ValueError):validate(bad)
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):validate(pd.concat([self.df,self.df.iloc[:1]]))
    def test_quantum_kernel(self):
        x=np.random.default_rng(1).uniform(0,np.pi,(12,4));psi=statevectors(x);k=kernel(x,x)
        np.testing.assert_allclose(np.linalg.norm(psi,axis=1),1,atol=1e-12)
        np.testing.assert_allclose(np.diag(k),1,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-10)
    def test_cci(self):
        self.assertAlmostEqual(float(cci(1,1)),1,places=6)
        self.assertEqual(float(cci(1,0)),0)
    def test_fuzzy_bounds(self):
        for x in np.linspace(0,1,8):
            for y in np.linspace(0,1,8):self.assertTrue(0<=fuzzy(x,y)<=2)
        self.assertLess(fuzzy(0,0),fuzzy(1,1))
    def test_demo_is_labelled(self):self.assertEqual(set(self.df.provenance),{'synthetic_demo'})
    def test_same_trial_future_target(self):
        for _,g in self.df.groupby(['subject_id','trial_id']):
            np.testing.assert_allclose(g.elbow_future_deg.to_numpy()[:-1],g.elbow_deg.to_numpy()[1:])
if __name__=='__main__':unittest.main()
