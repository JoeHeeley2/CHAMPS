from VEHICLe_read import read_mol_from_smiles
from file_generation import xyzcoords_from_type_array
from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import date
import numpy as np
from tqdm import tqdm
import os

def write_gaussian_command_with_regid(Regid, chkfilename, modredundant=False):

    '''
    Bad solution to a minor problem. Writes the gaussian command line for generation of .com files
    :param Regid:
    :param chkfilename:
    :param modredundant:
    :return:
    '''
    with open('Gaussian_command.txt', 'w') as p:
        p.write('%chk=' + chkfilename)
        p.write('\n')

        if modredundant:
            p.write('# opt=modredundant wb97xd/6-31g(d)')
        else:
            p.write('# opt wb97xd/6-31g(d)')

        p.write('\n')
        p.write('\n')
        p.write(Regid)
        p.write('\n')
        p.write('\n')
        p.write('0 1')

def com_from_xyz_coords(indexfilename, comfilename):

    '''
    Writes comfile from the xyz coords writen in indexfilename
    :param indexfilename: name of index file
    :param comfilename: name of output .com file
    :return:
    '''

    with open(comfilename, 'w') as fn:
        with open('Gaussian_command.txt') as fp:
            with open(indexfilename) as fd:

                for gaussline in fp:
                    print(gaussline.strip('\n'), file=fn)

                for indexline in fd:
                    split = indexline.split(' ')
                    print(split[0].strip('\n'), indexline[4:].strip('\n').lstrip(), file=fn)

def write_indexfile(dataframe):

    '''
    another botched solution to an easy-to-solve problem. Index files contain xyz coords and atom indexes
    for written molecules, I think it was to number the anions but it was stupid and convoluted. anyway,
    the code works as long as you dont try and understand why I did any of this.

    :param dataframe: VEHICLe subset containing structure to write
    :return:
    '''

    bad_ids = []

    smiles = dataframe['Smiles']
    for a in tqdm(smiles):
        row = dataframe[dataframe.Smiles == a]
        Regid = row.Regid.item()
        indexfilename = 'txt_files/' + str(Regid) + '_indexing.txt'

        mol = read_mol_from_smiles(a)
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)

        AllChem.MMFFOptimizeMolecule(molH)
        conf = molH.GetConformer()
        coordinates = conf.GetPositions()
        xyz_list = coordinates.tolist()

        coords_2d = AllChem.Compute2DCoords(molH)
        type_array = np.zeros(molH.GetNumAtoms(), dtype=np.int32)
        index_list = []

        for j, atoms in enumerate(molH.GetAtoms()):
            type_array[j] = atoms.GetAtomicNum()
            index = atoms.GetIdx()
            index_list.append(index)

        match_coords = zip(type_array, index_list, xyz_list)

        xyzcoords_from_type_array(indexfilename, match_coords)

def basehet_from_smiles(dataframe, workflow=False):

    '''
    writes .com file of heterocycle structure for geometry optimisation.
    :param dataframe: VEHICLe subset containing structures to write
    :param workflow: Bool, if False you will need to manually append blank lines to the end of the file
    :return:
    '''

    if not workflow:
        print('PLEASE NOTE: You are using neutral_generation.basehet_from_smiles outside of '
              'workflow.comfile_generation_workflow you need to append blank lines to all generated comfiles')

    regid = dataframe['Regid']
    todays_date = date.today()
    write_indexfile(dataframe)

    for i in tqdm(regid):

        indexfilename = 'txt_files/' + str(i) + '_indexing.txt'
        comfilename = 'comfiles/' + str(i) + '_' + str(todays_date) + '_wb97xd_631gd_opt.com'
        chkfilename = str(i) + '_' + str(todays_date) + '_wb97xd-631gd_opt.chk'

        write_gaussian_command_with_regid(i, chkfilename)
        com_from_xyz_coords(indexfilename, comfilename)






