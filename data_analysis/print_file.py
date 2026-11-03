from rdkit import Chem
from rdkit.Chem import Draw

def print_to_csv(dataframe, filename):
    with open (filename +'.csv', 'w') as f:
        print(dataframe.to_csv(sep=','), file=f)

def print_smiles_to_gridimage(dataframe, outname, n=None):
    print('Have you got the right criteria?')
    print('Have you given the right value for n?')
    mol_list = []
    smiles=dataframe['Smiles']
    ids = dataframe['Regid']
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        mol_list.append(mol)

    if n is None:
        n = round((len(mol_list)/3))
        slice_list = [n, 2 * n, 3 * n]

    else:
        slice_list = [n]

    for j, k in enumerate(slice_list):
        mol_slice = mol_list[(k - n):k]
        id_slice = ids.values.tolist()[(k - n):k]
        img = Draw.MolsToGridImage(mol_slice, legends=id_slice, molsPerRow=6)
        img.save(outname + str(j + 1) + '.png')
