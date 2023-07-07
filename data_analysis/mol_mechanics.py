from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from mol_translator import aemol
from mol_translator.imp_converter import dataframe_write as dfw

def write_mm_structures(vehicle_df):

    '''
    Think this is the same function as in imp_core, whyyyyy
    :param vehicle_df:
    :return:
    '''

    exp = ['S1072', 'S1080', 'S1863', 'S1869', 'S1873', 'S5703', 'S5704', 'S5706', 'S5707', 'S5853', 'S5854', 'S5857',
           'S6018', 'S1875', 'S291', 'S8063', 'S54', 'S7972']

    exp_df = vehicle_df[vehicle_df['Regid'].isin(exp)]

    mols = [Chem.MolFromSmiles(x) for x in exp_df.Smiles.values]
    ids = exp_df['Regid']
    amols = []

    for i, m in zip(ids, mols):

        molH = Chem.AddHs(m)
        Chem.AllChem.EmbedMolecule(molH)

        AllChem.MMFFOptimizeMolecule(molH)

        amol = aemol(molid=i)
        amol.from_rdkit(molH)
        amols.append(amol)

    atom_df = dfw.make_atom_df(amols)
    pair_df = dfw.make_pair_df(amols)

    atom_df.to_csv('dj1_exp_MM_atom.csv')
    pair_df.to_csv('dj1_exp_MM_pair.csv')

df = pd.read_csv('dj1/dj1_clean.csv')

write_mm_structures(df)
