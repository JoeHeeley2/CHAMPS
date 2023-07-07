import sys
import glob
from mol_translator import aemol
from mol_translator.structure.find_paths import pybmol_find_paths, pybmol_find_all_paths
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from random import randint
from sklearn.model_selection import StratifiedShuffleSplit
from impression_input import exclude_structures_imp_dfs
import os
import shutil
import copy

def first200_sampling(neutral_logfiles):

    '''
    Orders structures in the dj1 dataset according to Regid and samples the first 200 structures
    for model training.
    :param neutral_logfiles: list of optimised aromatic heterocycle .log files (no anions or bromines!)
    :return:
    '''

    training_set = neutral_logfiles[0:200]
    test_set = []

    for i in neutral_logfiles:
        if i not in training_set:
            test_set.append(i)

    for j in test_set:
        if j in training_set:
            raise ValueError('Same molecules in the test and training set')

    print('training set contains: ', len(training_set), 'molecules')
    print('test set contains: ', len(test_set), 'molecules')

    return training_set, test_set


def every2nd_sampling(neutral_logfiles):

    '''
    Arranges structures in order of regid and samples every 2nd structure into the training set.
    The remaining structures can then be used in the test set.
    :param neutral_logfiles: list of neutral .log files that contains optimised structures
    :return:
    '''

    training_set = sorted(neutral_logfiles, key=str)[0::2]
    test_set = [x for x in neutral_logfiles if x not in training_set]

    print('training set contains: ', len(training_set), 'molecules')
    print('test set contains: ', len(test_set), 'molecules')

    return training_set, test_set

def random_sample(sample_pool, length):
    '''
    Randomly samples n structures where n = length.
    :param sample_pool: a list of regids you want to sample from
    :param length: how many structures you want to sample
    :return: a list of .log files that have been randomly sampled
    '''

    sampled_structures = []
    pbar = tqdm(total=length)

    while len(sampled_structures) < length:
        value = randint(0, len(sample_pool) - 1)
        select = sample_pool[value]
        sampled_structures.append(select) if select not in sampled_structures else sampled_structures
        pbar.update(1)

    print('Sampled', len(sampled_structures), 'structures')

    return sampled_structures

def get_tm_df(training_structures, vehicle_df):
    '''
    Should return a dataframe of rdkit fingerprints and an array of fingerprint similarity
    :param training_structures: should be a list of regids you want to sample from
    :param VEHICLe_subset: should contain the SMILES strings for all your regids
                (you can use either the subset you're working from or just the full VEHICLe database)

    :return: tm_df (dataframe of molecular fingerprints)
    :return: tm_array (fingerprint similarity array)
    '''

    names = []
    ecfps = []
    for j in tqdm(training_structures):
        smiles = vehicle_df.loc[vehicle_df['Regid'] == j].Smiles.item()
        mol = Chem.MolFromSmiles(smiles)
        molH = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(molH, radius=2, nBits=2048)
        names.append(j)
        ecfps.append(fp)

    tm_array = np.memmap("fp_arr.dat", dtype='float32', mode='w+', shape=(len(names), len(names)))
    for i1, (id1, fp1) in tqdm(enumerate(zip(names, ecfps))):
        for i2, (id2, fp2) in enumerate(zip(names, ecfps)):
            tm_array[i1][i2] = DataStructs.FingerprintSimilarity(fp1, fp2)

    tm_df = pd.DataFrame(list(zip(names, ecfps)), columns=['molecule_name', 'ecfp'])

    return tm_df, tm_array

def get_fps_ids(tm_df, tm_array, outname, num=100, write=False):
    '''
    samples the n most diverse structures from the dataset, where n = num. Uses tanimoto
    similarity.
    :param tm_df: output [0] from get_tm_df
    :param tm_array: output [1] from get_tm_df
    :param num: number of structures you want to sample

    :return: list of len(num) of structures to sample,
                also writes the list as a .txt file if write=True
    '''
    pbar = tqdm(total=num)
    selected = list(np.random.choice(tm_df.molecule_name.unique(), 10))

    for molid in selected:
        pbar.update(1)

    while len(selected) < num:

        selected_ids = [x in selected for x in tm_df.molecule_name.values]
        big_d = 1000000.0
        to_pick = ''
        tm_array = np.memmap("fp_arr.dat", dtype='float32', mode='r',
                             shape=(len(tm_df.molecule_name), len(tm_df.molecule_name)))

        for molid in tm_df.molecule_name.unique():
            if molid in selected:
                continue
            id1 = tm_df.loc[(tm_df.molecule_name == molid)].index.values[0]
            distance = np.sum(tm_array[id1][selected_ids])
            if distance < big_d:
                big_d = distance
                to_pick = molid

        pbar.update(1)
        selected.append(to_pick)

    #d = dict(zip(molids, dist))
    #dsort = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    #selected = selected + list(dsort.keys())[:num]

    if write:
        with open(outname + str(num) + '.txt', 'w') as f:
            for j, molid in enumerate(selected):
                string = "{0:<10d}\t,\t{1:<10s}".format(j, molid)
                print(molid, file=f)

    return selected

def stratified_sampling_for_dj1_point2(atom, pair, outname):
    '''
    Stratified sampling for dj1_point2 dataset.

    NOTE: this has only been written for the HET ONY subset of dj1_point2 and will need expanding when the chembl set is added.
    Will take the atom df and do an 80:20 train:test split ensuring equal representation of each substitution in the datasets.
    :param atom: dj1_point2 atom dataframe
    :param pair: dj1_point2 pair dataframe
    :return:
    '''

    subs = []

    for i in atom['molecule_name'].unique():
        if '-' in i:
            c = i.split('-')[-1]
            subs.append(c)

        elif 'CHEMBL' in i:
            subs.append('chembl')

        else:
            subs.append('None')

    class_df = pd.DataFrame({'Regid': atom['molecule_name'].unique(),
                             'class': subs})

    sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)

    for train_index, test_index in sss.split(class_df, subs):
        strat_trainids = class_df.loc[train_index]['Regid'].values
        strat_testids = class_df.loc[test_index]['Regid'].values

    exclude_structures_imp_dfs(strat_testids, atom, pair, outname=outname + '_strat_train')
    exclude_structures_imp_dfs(strat_trainids, atom, pair, outname=outname + '_strat_test')
