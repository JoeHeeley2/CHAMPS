import numpy as np
import pandas as pd
from mol_translator.imp_converter import dataframe_write as dfw
from mol_translator.aemol import aemol
from mol_translator.structure.structure_write import write_mol_tosdf
from tqdm import tqdm
import os
import random

def file_to_aemol(logfiles, write=False, ftype='log'):

    '''
    Convert structures within files to AutoEnrich molecules. Defaults to Gaussian log files
    :param logfiles: files to convert
    :param write: Bool, if True AEmol structures are written as .sdf files
    :param ftype: filetype, defaults to log but others are available. see MolTranslator documentation
    :return: list of AEmol objects
    '''

    array = np.unique(np.array(logfiles))
    amols = []

    for file in tqdm(list(array)):
        x = file.split('\\')[-1]
        print(x)
        p = x.split('_')[0]
        print(p)

        outfile = 'aemols/' + str(p) + '_AEMOL.sdf'

        amol = aemol(p, filepath=file)
        amol.from_file(file, ftype=ftype)
        amol.prop_fromfile(file, 'g16', 'scf')

        assert amol.mol_properties['energy'] < 1000000, print(amol.mol_properties)

        amol.get_bonds()
        amol.get_path_lengths()

        if amol not in amols:
            amols.append(amol)
        else:
            print('amol error', amol.info['molid'])
        if write:
            write_mol_tosdf(amol, outfile)

    return amols

def xyzfile_to_aemol(xyzfiles, write=False):

    '''
    Supposedly converts xyz file to AEmoles but I dont think it works
    :param xyzfiles: list of xyzfiles
    :param write: Bool, saves mols as .sdf files if True
    :return: list of AEmols objects
    '''

    amols = []
    for file in tqdm(xyzfiles):
        x = file.split('\\')
        p = x[1].split('_')[0]
        try:
            outfile = 'aemols/' + str(p) + '_AEMOL.sdf'
            if os.path.isfile(outfile):

                amol = aemol(p)
                amol.from_file(file, ftype='xyz')

                assert amol.mol_properties['energy'] < 1000000, print(amol.mol_properties)

                amol.get_bonds()
                amol.get_path_lengths()
        finally:
            amols.append(amol)
            if write:
                write_mol_tosdf(outfile, amol)
    return amols

def write_imp_input(aemols, fulldata_df, outname, write=True):

    '''
    Writes input for machine learning models from the output of property_calculate.calculate_properties()
    :param aemols: list of AutoEnrich molecule objects
    :param fulldata_df: output dataframe from property_calculate.calculate_properties() function.
    :param outname: output filemane for all generated files
    :param write: Bool, writes atom and pair dataframs to .csv if True
    :return: atom_dataframe and pair_dataframe which are inputs for IMPRESSION and CHAMPS.
    '''

    atom_df = dfw.make_atom_df(aemols)
    pair_df = dfw.make_pair_df(aemols)

    regid = list(set(atom_df.molecule_name.unique()))

    props = np.zeros(len(atom_df))

    for j in regid:
        mol_prop = fulldata_df[fulldata_df['Regid'].str.fullmatch(j)]
        acidities = mol_prop.filter(regex='acidity').values
        elec_affs = mol_prop.filter(regex='electrophile_affinity').values
        mol_df = atom_df[atom_df['molecule_name'].str.fullmatch(j)]
        catom_index = mol_df[mol_df['typestr'].str.fullmatch('C')].atom_index
        hatom_index = mol_df[mol_df['typestr'].str.fullmatch('H')].atom_index
        # because of my stupid numbering I had to identify the non-quaternary carbons using the connectivity array
        # Did this by finding the positions of the H's and seeing if the atom in question was attached
        carbons = []
        hydrogens = []

        for ind in catom_index.values:
            connected_hs = []
            c_conn = mol_df.iloc[ind].conn.tolist()
            for i in hatom_index.values:
                if c_conn[i] == 1:
                    connected_hs.append(i)

            if len(connected_hs) == 1:
                carbons.append(ind)
                hydrogens.extend(connected_hs)

        for a in acidities:
                for u,v, in zip(hydrogens, a):
                    hdf_index = mol_df.loc[mol_df['atom_index'] == u].index.values
                    np.put(props, [hdf_index], [v])

        for e in elec_affs:
                for x,y in zip(carbons, e):
                    cdf_index = mol_df.loc[mol_df['atom_index'] == x].index.values
                    np.put(props, [cdf_index], [y])

    atom_df['shift'] = props
    final_atom_df = atom_df.fillna(0)

    if write:

        with open(outname + '_atom_df.csv', 'w') as p:
            print(final_atom_df.to_csv(sep=','), file=p)

        with open(outname + '_pair_df.csv', 'w') as f:
            print(pair_df.to_csv(sep=','), file=f)

    return final_atom_df, pair_df

def djr_classifier_input(vehicle_dataframe, relative_properties_dataframe, outname,
                     djr_lim=1.21, acid_lim=0.67, nuc_lim=0.84, write=True):

    '''
    Writes a dataframe for the input of a CHAMPS classification machine. Will set unreactive structures to 0, acidic structures
    to 1 and nucleophilic structures to 2. limits are experimentally determined and correspond to the quadrant model

    :param vehicle_dataframe: The VEHICLe dataframe - must contain same structures as calculate_properties_dataframe
    :param calculate_properties_dataframe: Output from calculate_properties
    :param outname: Name to save the output dataframe under
    :param write: Bool: saves if True
    :param djr_lim: limit above which acidic activation is seen (experimentally determined)
    :return:
    '''

    regid = relative_properties_dataframe['Regid']

    class_dataframe = pd.DataFrame()

    for i in regid:

        acidities = {}
        elec_affs = {}

        df = relative_properties_dataframe.loc[relative_properties_dataframe['Regid'] == i]

        for j in df.columns:
            if 'anion' in j:
                if df[j].values != 0:
                    acidities[j] = float(df[j].values)

            elif 'bromine' in j:
                elec_affs[j] = float(df[j].values)

        most_acidic = round(max(acidities.values()), 2)
        most_nuc = round(max(elec_affs.values()), 2)

        djr = round(most_acidic / most_nuc, 2)

        activation = -404.404

        if most_acidic < acid_lim and most_nuc < nuc_lim:
            activation = 2

        elif most_acidic > acid_lim and most_nuc < nuc_lim:
            activation = 1

        elif most_acidic > acid_lim and most_nuc > nuc_lim:
            if djr > djr_lim:
                activation = 1

            else:
                activation = 0

        else:
            activation = 0

        if activation == -404.404:
            print(i, most_acidic, most_nuc, djr)

        data = pd.DataFrame({'Regid': i, 'Smiles': vehicle_dataframe[vehicle_dataframe['Regid'] == i].Smiles.values[0], 'djr': djr,
                'class': activation}, index=[0])

        class_dataframe = pd.concat([class_dataframe, data])

    with open(outname +'.csv', 'w') as f:
        print(class_dataframe.to_csv(sep=','), file=f)

def donicity_classifier_input(ncs_atom, vehicle_df):

    '''
    why is this here?
    :param ncs_atom:
    :param vehicle_df:
    :return:
    '''

    ids = ncs_atom['molecule_name'].sort_values('molecule_name').unique()
    class_ids = []
    struct_class = []

    smiles = vehicle_df[vehicle_df['Regid'].isin(ids)].sort_values('Regid')

    for i in ids:

        smiles = vehicle_df[vehicle_df['Regid'] == i].Smiles.values[0]
        ncs_row_atom = ncs_atom[ncs_atom['molecule_name'] == i]

        n = ncs_row_atom[ncs_row_atom['typestr'] == 'N'].sort_values('shift', ascending=False)
        c = ncs_row_atom[ncs_row_atom['typestr'] == 'C'].sort_values('shift', ascending=False)
        h = ncs_row_atom[ncs_row_atom['typestr'] == 'H'].sort_values('shift')
        h = h[h['shift'] != 0]

def exclude_structures_imp_dfs(structures_to_exclude, atom_df, pair_df, outname, write=True):


    '''
    Function to remove specific structures from atom and pair dataframes.
    :param structures_to_exclude: list of structures to remove
    :param atom_df: atom_dataframe
    :param pair_df: pair_dataframe
    :param outname: output filename for resultant atom and pair dataframes
    :param write: Bool, if True atom and pair dataframes are written as .csv files
    :return:
    '''

    excluded_atom = atom_df.drop(atom_df.index[atom_df['molecule_name'].isin(structures_to_exclude)])
    excluded_pair = pair_df.drop(pair_df.index[pair_df['molecule_name'].isin(structures_to_exclude)])

    print('Removed:', len([x for x in atom_df['molecule_name'].unique() if x not in excluded_atom['molecule_name'].unique()]), 'structures')

    if write:
        with open(outname + '_atom_df.csv', 'w') as f:
            print(excluded_atom.to_csv(), file=f)

        with open(outname + '_pair_df.csv', 'w') as p:
            print(excluded_pair.to_csv(), file=p)

    return excluded_atom, excluded_pair

def exclude_structures_prop_dfs(structures_to_exclude, df1, df2, write=True):

    '''
    Removes structures from property dataframes. Not really sure why theres two here...
    :param structures_to_exclude: list of structures to remove
    :param df1:
    :param df2:
    :param write:
    :return:
    '''

    excluded_df1 = df1.drop(df1.index[df1['Regid'].isin(structures_to_exclude)])
    excluded_df2 = df2.drop(df2.index[df2['Regid'].isin(structures_to_exclude)])

    print('Removed:',
          len([x for x in df1['Regid'].unique() if x not in excluded_df1['Regid'].unique()]),
          'structures from df1')

    print('Removed:',
          len([x for x in df2['Regid'].unique() if x not in excluded_df2['Regid'].unique()]),
          'structures from df2')

    if write:
        print('you need to write the code to save the excluded dfs ya dingus')

    return excluded_df1, excluded_df2

def merge_datasets(dataset1_atom, dataset1_pair, dataset2_atom, dataset2_pair, outname, write=True):

    '''
    Merges two datasets and ensures no suplicates in the final dataframe.
    :param dataset1_atom: first atom df to merge
    :param dataset1_pair: first pair df to merge
    :param dataset2_atom: second atom df to merge
    :param dataset2_pair: second pair df to merge
    :param outname: output filename for all written files
    :param write: Bool, if True writes the merged dataframe as .csv file
    :return:
    '''

    #use the exclude_structures function to remove any overlapping structures between the two input dataframes
    # before concatenation
    structures_to_exclude = dataset1_atom['molecule_name'].unique()

    unique_dfs = exclude_structures_imp_dfs(structures_to_exclude, dataset2_atom, dataset2_pair, outname=None, write=False)

    join_atom = pd.concat([dataset1_atom, unique_dfs[0]], axis=0)
    join_pair = pd.concat([dataset1_pair, unique_dfs[1]], axis=0)

    if write:
        with open(outname + '_atom_df.csv', 'w') as f:
            print(join_atom.to_csv(), file=f)

        with open(outname + '_pair_df.csv', 'w') as p:
            print(join_pair.to_csv(), file=p)

    return join_atom, join_pair

def test_train_val_split(atom_df, pair_df, outname):

    '''
    Splits stom and pair dataframe with a random 60: 20: 20 train: test: val split
    :param atom_df: atom_df to split
    :param pair_df: pair_df to split
    :param outname: output filename for all written files
    :return: None
    '''

    molids = atom_df['molecule_name'].unique()

    random.shuffle(molids)

    tenpercent = round(len(molids) / 10)

    trainids = molids[:(tenpercent * 6)]

    valids = molids[(tenpercent * 6):(tenpercent * 8)]

    testids = molids[(tenpercent * 8):]

    train_atom_df = atom_df.drop(atom_df.index[atom_df['molecule_name'].isin(np.append(valids, testids))])
    train_pair_df = pair_df.drop(pair_df.index[pair_df['molecule_name'].isin(np.append(valids, testids))])

    val_atom_df = atom_df.drop(atom_df.index[atom_df['molecule_name'].isin(np.append(trainids, testids))])
    val_pair_df = pair_df.drop(pair_df.index[pair_df['molecule_name'].isin(np.append(trainids, testids))])

    test_atom_df = atom_df.drop(atom_df.index[atom_df['molecule_name'].isin(np.append(valids, trainids))])
    test_pair_df = pair_df.drop(pair_df.index[pair_df['molecule_name'].isin(np.append(valids, trainids))])

    with open(outname + '_TRAIN_ATOM.csv', 'w') as f:
        print(train_atom_df.to_csv(), file=f)

    with open(outname + '_TRAIN_PAIR.csv', 'w') as p:
        print(train_pair_df.to_csv(), file=p)

    with open(outname + '_VAL_ATOM.csv', 'w') as y:
        print(val_atom_df.to_csv(), file=y)

    with open(outname + '_VAL_PAIR.csv', 'w') as l:
        print(val_pair_df.to_csv(), file=l)

    with open(outname + '_TEST_ATOM.csv', 'w') as d:
        print(test_atom_df.to_csv(), file=d)

    with open(outname + '_TEST_PAIR.csv', 'w') as x:
        print(test_pair_df.to_csv(), file=x)


