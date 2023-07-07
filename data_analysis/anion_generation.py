import glob
import numpy as np
import rdkit.Chem.rdForceFieldHelpers
from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import date
from tqdm import tqdm

def anions_from_smiles(dataframe, workflow=False):

    '''
    Function writes gaussian input files for the calculations of relative exchange acidities
    of aromatic heterocycles starting from the SMILES string. It deprotonated each sp2 hybridised CH bond
    and writes a com file for the corresponding anion. Requires a directory of 'txt_files' and 'comfiles'.
    If the anion is generated next to a heteroatom-heteroatom bond, the heteroatom-heteroatom bond is frozen
    to prevet it from dissociating. Because of this, all sturctures are pre-optimised using molecular mechanics.

    There are easier ways to do this in probably half the time...

    :param dataframe: VEHICLe subset containing SMILES of structures you want to calculate acidities for
    :param workflow: Bool, necessary for modredundant calculations. if False, blank lines should be appended
                        at the end of the files
    :return: None, writes comfiles to 'comfiles/' directory
    '''

    if not workflow:
        print('PLEASE NOTE: If you are using neutral_generation.basehet_from_smiles outside of '
              'workflow.comfile_generation_workflow you need to append blank lines to all generated comfiles')

    regid = dataframe['Regid']

    for l in tqdm(regid):
        comfiles = glob.glob('comfiles/' + str(l) + '_*')
        anion_comfiles = [x for x in comfiles if 'bromine' not in x]
        neutral_comfiles = [y for y in anion_comfiles if 'anion' not in y]
        index_file = glob.glob('txt_files/' + str(l) + '_indexing.txt')
        H_index_list = []
        C_index_list = []
        C_numbering = []
        Het_index = []
        row = dataframe[dataframe.Regid == l]
        smiles = row.Smiles.item()

        mol = Chem.MolFromSmiles(smiles)
        molH = Chem.AddHs(mol)
        a = molH.GetAtoms()

        for j, atoms in enumerate(a):
            b = atoms.GetAtomicNum()
            d = atoms.GetIdx()
            if b == 1:
                e = atoms.GetNeighbors()
                for j, k in enumerate(e):
                    if k.GetAtomicNum() == 6:
                        if str(k.GetHybridization()) == 'SP2':
                            H_index_list.append(d)
                            C_index = k.GetIdx()
                            C_index_list.append(C_index)
                            for w in k.GetNeighbors():
                                if w.GetAtomicNum() != 6:
                                        for x in w.GetNeighbors():
                                            if x.GetAtomicNum() != 6:
                                                if x.GetAtomicNum() != 1:
                                                        hets = [w.GetIdx(), x.GetIdx(), d]
                                                        Het_index.append(hets)

        for x in range(len(C_index_list)):
            C_numbering.append(x + 1)


        match_indexes = zip(H_index_list, C_numbering)

        todays_date = str(date.today())

        for file in neutral_comfiles:
            split = file.split('\\')
            name = split[1].split('_')
            for m in index_file:
                for n in match_indexes:
                    with open(m) as p:
                        for line in p:
                            comindex = line[2] + line[3]
                            if float(n[0]) == float(comindex):
                                xyz = line[4:]
                                anion_comfilename = 'comfiles/' + name[0] + '_' + str(
                                    n[1]) + 'anion_' + todays_date + '_' + \
                                                    name[2] + '-' + name[3] + '_opt.com'

                                anion_chkfilename = name[0] + '_' + str(n[1]) + 'anion_' + todays_date + '_' + name[
                                    2] + '-' + name[3] + '_opt.chk'

                                with open(anion_comfilename, 'w') as q:
                                    with open(file) as f:
                                        for comline in f:
                                            com_xyz = comline[2:]

                                            if comline.startswith('%chk='):
                                                print('%chk=' + anion_chkfilename, file=q)

                                            elif '0 1\n' in comline:
                                                print('-1 1', file=q)

                                            elif str(xyz).lstrip() != com_xyz:
                                                if len(Het_index) > 0:
                                                    if comline.startswith('# opt'):
                                                        print('# opt=modredundant wb97xd/6-31g(d)', file=q)

                                                    else:
                                                        print(comline.strip('\n'), file=q)
                                                else:
                                                    print(comline.strip('\n'), file=q)

                                    if len(Het_index) > 0:
                                        print('\n'.strip('\n'), file=q)
                                        for h, k, v in Het_index:
                                            if v == n[0]:
                                                print('B', (h + 1), (k + 1), 'F', file=q)
