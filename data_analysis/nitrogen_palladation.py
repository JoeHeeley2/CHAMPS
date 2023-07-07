from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from file_generation import xyzcoords_from_type_array, type_dict

import numpy as np
from neutral_generation import write_gaussian_command_with_regid, com_from_xyz_coords
import os
from tqdm import tqdm
from file_generation import write_orca_file_for_directed_activation_opt_and_freq

def palladate_nitrogens(vehicle_dataframe):


    '''
    Function that palladates all basic nitrogens for the calculation of donicity metric. All non-NH nitrogens
    are coordinated to a [Pd(OAc)(SPHOS)]+ fragment and the xyz coords are extracted. Orca .inp file are
    then written into the 'directed_activation' directory. this is done for each basic nitrogen in the dataset

    :param vehicle_dataframe: VEHICLe_subset that contains structures to palladate.
    :return:
    '''

    Pd_frag = Chem.MolFromSmiles('CC1=[O+][Pd-2]([P+](C2CCCCC2)(C3=CC=CC=C3C4=C(OC)C=CC=C4OC)C5CCCCC5)O1')

    for id in tqdm(vehicle_dataframe['Regid']):

        row = vehicle_dataframe[vehicle_dataframe.Regid == id]
        smiles = row.Smiles.item()

        het = Chem.MolFromSmiles(smiles)
        het_h = Chem.AddHs(het)

        combo = Chem.CombineMols(Pd_frag, het_h)

        combo_mols = []
        N_idx = []

        for atom in combo.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if len(atom.GetNeighbors()) == 2:
                    edcombo = rdchem.EditableMol(combo)
                    N_idx.append(atom.GetIdx())
                    edcombo.AddBond(atom.GetIdx(), 3, order=rdchem.BondType.DATIVE)
                    combo_mols.append(Chem.AddHs(edcombo.GetMol()))

        for i, k in zip(combo_mols, N_idx):

            indexfilename = 'txt_files/' + str(k) + '-' + id + '.idx'
            inpfilename = 'directed_activation/' + str(k) + '-' + id + '_opt-freq_wb97xd.inp'
            xyzfilename = str(k) + '-' + id + '_opt-freq_wb97xd.xyz'

            AllChem.EmbedMolecule(i)
            AllChem.MMFFOptimizeMolecule(i)
            conf = i.GetConformer()
            coordinates = conf.GetPositions()
            xyz_list = np.array(coordinates.tolist(), dtype=float)

            coords_2d = AllChem.Compute2DCoords(i)
            type_array = np.zeros(i.GetNumAtoms(), dtype=np.int32)
            index_list = []

            for j, atoms in enumerate(i.GetAtoms()):
                type_array[j] = atoms.GetAtomicNum()
                index = atoms.GetIdx()
                index_list.append(index)

            match_coords = zip(type_array, index_list, xyz_list)

            xyzcoords_from_type_array(indexfilename, match_coords)
            write_orca_file_for_directed_activation_opt_and_freq(str(k) + '_' + id, indexfilename, inpfilename)

            os.remove(indexfilename)

def generate_basehet_input(amol):

    '''
    writes the netural heterocycle structure to an Orca .inp file for the calculation of the donicity metric.
    optimisation and frequency calculations are carried out at the wB97x-D3BJ/def2-TZVP level of theory.
    :param amol: AutoEnrich molecule object of structure to write
    :return:
    '''

    types = type_dict()

    structure = amol.info['molid']
    xyz = amol.structure['xyz']
    atoms = amol.structure['types']

    with open('directed_activation/basehets/' + structure + '-basehet_opt-freq_wb97xd.inp', 'w') as f:

        f.write('! wB97x-d3bj def2-TZVP CPCM(Toluene) xyzfile opt freq\n')
        f.write('%base "' + structure + 'OptFreq"\n')
        f.write('%maxcore 2000\n')
        f.write('%scf MaxIter 1500 end\n')
        f.write('%pal nprocs 2\n')
        f.write('end\n')
        f.write('\n')
        f.write('*xyz 0 1\n')

        for type, coord in zip(atoms, xyz):
            print(types[type], *list(coord), file=f)

        f.write('*\n')
        f.write('\n')
        f.write('$new_job\n')
        f.write('! wB97x-d3bj def2-TZVP CPCM(Toluene) printbasis\n')
        f.write('%base"' + structure + 'SinglePoint"\n')
        f.write('\n')
        f.write('*xyzfile 0 1 ' + structure + 'OptFreq.xyz')
        f.write('\n')

