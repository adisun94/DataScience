
import unittest
import pandas as pd
import sys

sys.path.append('../')

from Functions import ChemClean
from Functions import Descriptors
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors

def feature_reduction(rdkit_features):
    p=ChemClean.clean(rdkit_features)

    p.remove_nan().head()
    p.remove_unique().head()
    p.remove_columns_low_std(0.1).head()
    df_model=p.remove_corr_features(0.7)
   
    return df_model.shape


class TestFeatureEngineering(unittest.TestCase):
    
    def test_feature_reduction(self):
        rdkit_features=pd.read_csv('FeaturesForUnitTest-all.csv',index_col=0)
        actual = feature_reduction(rdkit_features)
        expected = pd.read_csv('FeaturesForUnitTest-cleaned.csv',index_col=0).shape
    
        self.assertEqual(actual, expected)
