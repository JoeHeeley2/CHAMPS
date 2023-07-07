from dscribe.descriptors import SOAP
from ase import Atoms
import pandas as pd
import numpy as np
from mol_translator.imp_converter.dataframe_read import read_df
from ase import Atoms
import dscribe.descriptors as desc
import datetime
from tqdm import tqdm

def gen_amols(atom_df, pair_df):

    amols = read_df(atom_df, pair_df)

    return amols

def type_dict(warning=True):

    if warning:
        print('Type Dict includes: H, C, N, O, S, F and Br ONLY! Please append it if you are including other atoms!'
              'See: dscribe_inputs.type_dict)')

    type_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 35: 'Br'}

    return type_dict

def add_soap_to_atom_df(atom_df, pair_df, amols=None):

    '''
    Adds the SOAP molecular representation to the atom dataframe
    :param atom_df: atom dataframe
    :param pair_df: pair_dataframe
    :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                    they are generated from the atom and pair dataframe
    :return:
    '''

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols=amols

    species = set()
    regids = []
    structures = []

    type_d = type_dict()

    for i in amols:
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]

        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])

    soap = desc.SOAP(species=species,
                periodic=False,
                rcut=5,
                nmax=8,
                lmax=8,
                average='outer',
                sparse=False)

    feat_vect = soap.create(structures, n_jobs=1)

    tmp = pd.DataFrame(zip(regids, feat_vect), columns=['regids', 'soap'])

    tmp_sort = tmp.sort_values('regids')
    tmp_sort.reset_index()

    atom_df_sort = atom_df.sort_values(['molecule_name', 'atom_index'])

    soap_list = []

    for lists in tmp_sort.soap:
        for item in lists:
            soap_list.append(item)

    #atom_df_sort['soap'] = soap_list

def add_acsf_to_atom_df(atom_df, pair_df, amols=None,):
    '''
        Adds the ACSF molecular representation to the atom dataframe
        :param atom_df: atom dataframe
        :param pair_df: pair_dataframe
        :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                        they are generated from the atom and pair dataframe
        :return: dataframe with added representation
        '''

    print('Setting ACSF feature length to 80. This will need to be changed molecules contain more than 5 atom types')

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []

    type_d = type_dict(warning=False)

    for i in amols:
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]
        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])

    acsf = desc.ACSF(
        species=species,
        rcut=7.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, 1]],
    )

    feat_vect = acsf.create(structures, n_jobs=10)

    tmp = pd.DataFrame(zip(regids, feat_vect), columns=['regids', 'acsf'])

    tmp_sort = tmp.sort_values('regids')
    tmp_sort.reset_index()

    atom_df_sort = atom_df.sort_values(['molecule_name', 'atom_index'])

    acsf_list = []

    for lists in tmp_sort.acsf:
        for item in lists:
            while len(item) < 80:
                item = np.append(item, [0])

            acsf_list.append(item)

    atom_df_sort['atomic_rep'] = acsf_list

    return atom_df_sort

def add_coulomb_matrix_to_atom_df(atom_df, pair_df, amols=None,):
    '''
            Adds the Coulomb Matrix molecular representation to the atom dataframe
            :param atom_df: atom dataframe
            :param pair_df: pair_dataframe
            :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                            they are generated from the atom and pair dataframe
            :return: dataframe with added representation
            '''

    print('Generating 28x28 Coulomb Matrices, see model.dscribe_inputs.add_coulomb_matrix_to_atom_df (line 133) to change')

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []

    type_d = type_dict(warning=False)

    for i in amols:
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]
        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])

    #if you change matrix size make sure you also resize the reshape on line 150
    cm = desc.CoulombMatrix(n_atoms_max=28)

    matrices = cm.create(structures)

    tmp = pd.DataFrame(zip(regids, matrices), columns=['regids', 'atomic_rep'])

    tmp_sort = tmp.sort_values('regids')
    tmp_sort.reset_index()

    atom_df_sort = atom_df.sort_values(['molecule_name', 'atom_index'])


    atom_cmatrix = []

    for i in tmp_sort['atomic_rep']:

        arr = i.reshape(-1, 28)

        for j in arr:
            if np.count_nonzero(j) > 0:
                atom_cmatrix.append(j.tolist())

    atom_df_sort['atomic_rep'] = atom_cmatrix

    return atom_df_sort

def add_local_many_body_tensor_to_atom_df(atom_df, pair_df, weight_func, k2_param_dict, k3_param_dict, amols=None):
    '''
            Adds the LMBT molecular representation to the atom dataframe
            :param atom_df: atom dataframe
            :param pair_df: pair_dataframe
            :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                            they are generated from the atom and pair dataframe
            :return: dataframe with added representation
            '''

    print('Adding LMBTR to atom_df...')
    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []
    xyz = []

    type_d = type_dict(warning=False)
    for i in tqdm(amols):
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]
        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])
        xyz.append(i.structure['xyz'].tolist())

    lmbtr = desc.LMBTR(
        species=species,
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": k2_param_dict['max'],
                     "n": k2_param_dict['n'],
                     "sigma": k2_param_dict['sigma']
                     },
            "weighting": {"function": weight_func,
                          "scale": 0.5, "threshold": 1e-3},
        },

        k3={
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": k3_param_dict['max'],
                     "n": k3_param_dict['n'],
                     "sigma": k3_param_dict['sigma']
                     },
            "weighting": {"function": weight_func,
                          "scale": 0.5, "threshold": 1e-3},
        },
        periodic=False,
        normalization="none",
    )

    feat_vect = lmbtr.create(structures)

    tmp = pd.DataFrame(zip(regids, feat_vect), columns=['regids', 'lmbtr'])

    tmp_sort = tmp.sort_values('regids')
    tmp_sort.reset_index()

    lmbtr_list = []

    for lists in tmp_sort.lmbtr:
        for item in lists:
            lmbtr_list.append(item.tolist())

    atom_df_sort = atom_df.sort_values(['molecule_name', 'atom_index'])
    atom_df_sort['atomic_rep'] = lmbtr_list

    return atom_df_sort

def add_many_body_tensor_to_class_df(atom_df, pair_df, class_df, amols=None,):
    '''
            Adds the MBT molecular representation to the class dataframe
            :param atom_df: atom dataframe
            :param pair_df: pair_dataframe
            :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                            they are generated from the atom and pair dataframe
            :return: dataframe with added representation
            '''

    print('Adding Many Body Tensor Representation to class_df...')

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []
    xyz = []

    type_d = type_dict(warning=False)
    for i in amols:
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]
        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])
        xyz.append(i.structure['xyz'].tolist())

    mbtr = desc.MBTR(
                    species=species,
                    k2={
                        "geometry": {"function": "distance"},
                        "grid": {"min": 0.4, "max": 8, "sigma": 0.1, "n": 200},
                    },
                    periodic=False,
                    flatten=True,
                    sparse=False,
                    normalization="l2_each",
                )

    feat_vect = np.array(mbtr.create(structures), dtype=np.float64)

    tmp = pd.DataFrame(zip(regids, feat_vect), columns=['Regid', 'mol_rep'])
    classes = []

    for reg in tmp['Regid'].values:
        classes.append(class_df.loc[class_df.Regid == reg]['class'].values[0])

    tmp['class'] = classes

    drop = tmp.dropna()

    if len(drop) != len(tmp):
        print('Found NaN in class_df, please check data before training')

    return drop

def add_coulomb_matrix_to_class_df(atom_df, pair_df, class_df, amols=None,):
    '''
                Adds the coulomb matrix molecular representation to the class dataframe
                :param atom_df: atom dataframe
                :param pair_df: pair_dataframe
                :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                                they are generated from the atom and pair dataframe
                :return: dataframe with added representation
                '''

    print('Generating 18x18 Coulomb Matrices, see model.dscribe_inputs.add_coulomb_matrix_to_atom_df (line 303) to change'
          'Also adding shifts to each line of the CM, you will have to rewrite the code to take this out')

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []
    matrices = []


    type_d = type_dict(warning=False)

    for i in tqdm(amols):
        molrow = atom_df[atom_df['molecule_name'] == i.info['molid']]
        molshift = molrow['shift'].values
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]
        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        #structures.append(structure)
        regids.append(i.info['molid'])

        cm = desc.CoulombMatrix(n_atoms_max=18)

        matrix = cm.create(structure)

        molshift = np.append(molshift, np.zeros((18 - len(list(molshift)))))

        split = np.split(matrix, 18)

        fnl = np.empty(0)
        for mx, shft in zip(split, molshift):
            com = np.append(mx, shft)
            fnl = np.concatenate([fnl, com])

        matrices.append(fnl)

    tmp = pd.DataFrame(zip(regids, matrices), columns=['Regid', 'mol_rep'])
    classes = []

    for reg in tmp['Regid'].values:
        classes.append(class_df.loc[class_df.Regid == reg]['class'].values[0])

    tmp['class'] = classes

    drop = tmp.dropna()

    if len(drop) != len(tmp):
        print('Found NaN in class_df, please check data before training')

    return drop

def add_soap_to_class_df(atom_df, pair_df, class_df, amols=None):
    '''
                Adds the SOAP molecular representation to the class dataframe
                :param atom_df: atom dataframe
                :param pair_df: pair_dataframe
                :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                                they are generated from the atom and pair dataframe
                :return: dataframe with added representation
                '''

    print('Adding SOAP to class_df...')

    if amols is None:
        amols = gen_amols(atom_df, pair_df)

    else:
        amols = amols

    species = set()
    regids = []
    structures = []

    type_d = type_dict(warning=False)

    for i in amols:
        atom_symbols = [x if x not in type_d else type_d[x] for x in i.structure['types']]

        structure = Atoms(symbols=atom_symbols, positions=i.structure['xyz'])
        species.update(structure.get_chemical_symbols())
        structures.append(structure)
        regids.append(i.info['molid'])

    soap = desc.SOAP(species=species,
                periodic=False,
                rcut=5,
                nmax=8,
                lmax=8,
                average='outer',
                sparse=False)

    feat_vect = soap.create(structures, n_jobs=1)

    tmp = pd.DataFrame(zip(regids, feat_vect), columns=['Regid', 'mol_rep'])
    classes = []

    for reg in tmp['Regid'].values:
        classes.append(class_df.loc[class_df.Regid == reg]['class'].values[0])

    tmp['class'] = classes

    drop = tmp.dropna()

    if len(drop) != len(tmp):
        print('Found NaN in class_df, please check data before training')

    return drop

def add_DFT_input(atom_df, class_df, type='shift'):
    '''
                Adds the DFT molecular representation to the class dataframe
                :param atom_df: atom dataframe
                :param pair_df: pair_dataframe
                :param amols: list of AutoEnrich Molecule objects. If none is supplied,
                                they are generated from the atom and pair dataframe
                :return: dataframe with added representation
                '''

    print('adding DFT representation...')

    regids = atom_df['molecule_name'].unique()
    shifts = []
    classes = []

    for i in regids:
        molshift = atom_df[atom_df['molecule_name'] == i][type].values
        app = np.append(np.array(molshift), np.zeros((18 - len(list(molshift)))))
        shifts.append(app)
        classes.append(class_df[class_df['Regid'] == i]['class'].values[0])

    class_df = pd.DataFrame({'Regid': regids, 'mol_rep': shifts, 'class': classes})

    return class_df

