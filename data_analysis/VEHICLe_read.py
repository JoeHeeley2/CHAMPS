from rdkit import Chem
import pandas as pd

def read_csv_to_dataframe(csv_file):
    df=pd.read_csv(csv_file)
    return df

def read_list_of_smiles(dataframe):
    list_of_smiles=list(dataframe['Smiles'])
    return list_of_smiles

def read_mol_from_smiles(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return mol

def read_list_of_mol_objects(dataframe):
    list_of_smiles = list(dataframe['Smiles'])
    list_of_mol_objects = []

    for i in list_of_smiles:
        mol_object = Chem.MolFromSmiles(i)
        list_of_mol_objects.append(mol_object)

    return list_of_mol_objects

