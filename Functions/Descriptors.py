import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd

class desc:
    """Class to generate structural descriptors/fingerprints for molecules
    
    .. attribute:: smiles

       Array of SMILES strings corresponding to the molecules

    """

    def __init__(self,smile):
        """
        Constructor method to create array with SMILES strings.eliminate features whose standard deviation is less than a threshold t, set to a default value of 0.3.

        Args:
            smile: Input SMILES strings

        """

        self.smiles=smile
        self.Mol_descriptors=[]

    def RDkit_descriptors(self):
        """
        Method to generate 208 structural descriptors/fingerprints for moleculeseliminate features whose standard deviation is less than a threshold t, set to a default value of 0.3.

        Returns:
        DataFrame with 208 features for each entry.

        """

        mols = [Chem.MolFromSmiles(i) for i in self.smiles] 
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        desc_names = calc.GetDescriptorNames()

        for mol in mols:
        # add hydrogens to molecules
#         mol=Chem.AddHs(mol)
        # Calculate all 208 descriptors for each molecule
            descriptors = calc.CalcDescriptors(mol)
            self.Mol_descriptors.append(descriptors)

        return pd.DataFrame(self.Mol_descriptors, columns=desc_names)
