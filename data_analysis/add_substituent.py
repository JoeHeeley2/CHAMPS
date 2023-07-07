from random import randint
import pandas as pd
from rdkit import Chem as chm
from workflow import comfile_generation_workflow
from sampling import random_sample

def frag_type_array():

    frag_type_array = {'C': 'Methylated',
                       'F': 'Fluorinated',
                       '[Pd]': 'Palladated',
                       '[#6](-[#6])=[#8]': 'Acetylated',
                       '[#6]-[#6]': 'Et',
                       '[#6]-[#6]-[#6]': 'nPr',
                       '[#6]-[#6]-[#6]-[#6]': 'nBu',
                       '[#6](-[#6])(-[#6])-[#6]': 'tBu',
                       '[#6](-[#6])-[#6]': 'iPr',
                       'CF': 'CFH2',
                       'C(F)F': 'CF2H',
                       'C(F)(F)F': 'CF3',
                       'C1=CC=CC=C1': 'Ph',
                       '[C]#N': 'CN',
                       'C1CCCC1': 'CyPent',
                       'C1CCC1': 'CyBut',
                       'CO': 'C2OH',
                       'COC': 'CH2OMe',
                       'CN': 'CH2NH2',
                       'CN(C)(C)': 'CH2NMe2',
                       'C(O)=O': 'CO2H',
                       'C(N)=O': 'CONH2',
                       'C(NC)=O': 'CONHMe',
                       'N1CCCCC1': 'Npiperidine',
                       'O': 'OH',
                       'OC': 'OMe',
                       'OC(C)(C)': 'OiPr',
                       'OC(F)(F)F': 'OCF3',
                       'OC(F)F': 'OCF2H',
                       '[N+](=O)[O-]': 'NO2',
                       'N': 'NH2',
                       'N1CCCC1': 'Npyrrolidine',
                       'N1CCNCC1': 'Npiperazine',
                       'N1CCOCC1': 'Nmorpholine',
                       'NC(=O)C': 'NHCOMe',
                       'C1CCNCC1': 'piperidineNH'}

    return frag_type_array

def add_substituent(dataframe, outname, frag_smiles='C', write=True, generate_com=True):

    '''
    Function to decorate heterocycles with pharmaceutically relevant substituents.
    Chooses 50 random structures from sampe pool and changes hydrogens randomly to substituent.
    Defaults to methylation.

    :param dataframe: VEHICLe_subset for decoration
    :param outname: filename for saved file.
    :param frag_smiles: SMILES string of the fragment to add (see frag_type_array)
    :param write:
    :param generate_com: Bool, if True comfiles for DFT property calculations are written
    :return: None
    '''

    sub_array = frag_type_array()

    rdmol = chm.MolFromSmiles(frag_smiles)
    smart = chm.MolToSmarts(rdmol)

    samp = random_sample(dataframe['Regid'], length=50)

    substituted_dataframe = pd.DataFrame(columns=['Regid', 'Smiles'])

    for i in samp:
        smiles = dataframe[dataframe['Regid'] == i].Smiles.values
        mol = chm.MolFromSmiles(smiles[0])
        molh = chm.AddHs(mol)
        h_frag = chm.MolFromSmarts('[H]')
        replace_frag = chm.MolFromSmarts(smart)

        mod_mol = chm.ReplaceSubstructs(molh, h_frag, replace_frag)

        structure_select = randint(0, len(mod_mol) - 1)

        structureh = mod_mol[structure_select]
        structure = chm.RemoveHs(structureh)

        sub_type = sub_array.get(frag_smiles)

        print(i + '-' + sub_type, chm.MolToSmiles(structure))

        sub_row = pd.DataFrame({'Regid': i + '-' + sub_type, 'Smiles': chm.MolToSmiles(structure)}, index=[0])

        substituted_dataframe = pd.concat([substituted_dataframe, sub_row])

    if write:
        with open(outname, 'w') as f:
            print(substituted_dataframe.to_csv(), file=f)

    if generate_com:
        comfile_generation_workflow(substituted_dataframe)
