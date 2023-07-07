import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import rdkit.Chem

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

def dj1_point2_filter(chembl_data, outname):

    '''
    CHEMBL filter that extracts compounds containing N-heteroarenes with less than 50 Heavy atoms.
    Amino Acid structures are removed also. Structures are saved as a .csv file with the CHEMBL IDs and SMILES
    :param chembl_data: the CHEMBL dataset as a dataframe
    :param outname: filename for saved files
    :return:
    '''

    regid = []
    smiles = []
    good_mol_atoms = [1, 6, 7, 8, 9]
    good_aromatic_atoms = [1, 6, 7]
    good_mol = False
    amino_acid = Chem.MolFromSmiles('NCC(NCC(NCC(N)=O)=O)=O')

    with open(chembl_data, 'r') as f:
        for line in tqdm(f.readlines()):
            linsplit = line.split()

            if 'canonical_smiles' not in linsplit[1]:
                rdmol = Chem.MolFromSmiles(linsplit[1])

                try:
                    atoms = rdmol.GetAtoms()

                except AttributeError:
                    print('Error:', linsplit[0], linsplit[1], 'Not Recognized')

                else:
                    if len(rdmol.GetAtoms()) < 51:

                        if not rdmol.HasSubstructMatch(amino_acid):

                            for molatom in atoms:
                                if molatom.GetAtomicNum() not in good_mol_atoms:
                                    break

                            else:
                                hmol = Chem.AddHs(rdmol)
                                ring = hmol.GetRingInfo()
                                ring_bonds = ring.BondRings()
                                ring_atoms = ring.AtomRings()

                                for atoms, bonds in zip(ring_atoms, ring_bonds):
                                    ch = False
                                    ring_atomtype = []
                                    if isRingAromatic(hmol, bonds):
                                        for j in atoms:
                                            atom = hmol.GetAtomWithIdx(j)
                                            num = atom.GetAtomicNum()
                                            if num not in good_aromatic_atoms:
                                                break

                                            else:
                                                ring_atomtype.append(num)
                                                if num == 6:
                                                    for n in atom.GetNeighbors():
                                                        if n.GetAtomicNum() == 1:
                                                            ch = True
                                    if ch:
                                        if 7 in ring_atomtype:
                                            if linsplit[1] not in smiles:
                                                regid.append(linsplit[0])
                                                smiles.append(linsplit[1])

    data = pd.DataFrame({'Regid': regid, 'Smiles': smiles})

    with open(outname + '.csv', 'w') as p:
        print(data.to_csv(), file=p)
