import pandas as pd
import glob
from tqdm import tqdm

def opt_check(file, string_to_search):
    with open(file, 'r') as f:
        for line in f:
            if string_to_search in line:
                return True
    return False

def orca_calculation_check(outfile_location, outname):

    '''
    Checks for failed calculations or imaginary frequencies in orca .out files for donicity
    calculation.
    :param outfile_location: path to directory which contains orca .out files
    :param outname: name to save file that stores failed and/or imaginary runs
    :return: list of failed runs, list of runs with imaginary freqs, list of successful runs
    '''


    failed_runs = []
    imgry_freq_files = []

    files = glob.glob(outfile_location + '/*.out')

    for i in tqdm(files):

        if not opt_check(i, '****ORCA TERMINATED NORMALLY****'):
            failed_runs.append(i)

        #freqs = []
        with open(i, 'r') as f:
            for line in f:
                if 'freq.' in line.rstrip():
                    freq = line.split()
                    if float(freq[1]) < 0:
                        print('Imaginary Frequency Detected:', i)
                        imgry_freq_files.append(i)
                    #freqs.append(freq)

    print('Found: ', len(failed_runs), 'Failed Runs')
    print('Found: ', len(imgry_freq_files), 'Imaginary Frequencies')

    with open(outname + '_failed_runs.txt', 'w') as p:
        for run in failed_runs:
            print(run.split('\\')[-1], file=p)

    successful_runs = [x for x in files if x not in failed_runs or imgry_freq_files]

    print('Found: ', len(successful_runs), 'Successful Runs')

    return failed_runs, imgry_freq_files, successful_runs


def calculate_Gcorrected(outfile_location, outname):

    '''
    Function to extract deltaG data from orca calculations. Such calculations should be the output of the
    palladate_nitrogens() function. This function only outputs the RAW data, and to calculate donicity,
    must be used as a precursor to calculate_deltaG_exch()

    :param outfile_location: path to directory containing all orca .out files
    :param outname: output filename for raw data .csv file
    :return:
    '''

    orca_calculation_check(outfile_location, outname)

    print('ADDING SOLVENT CORRECTION OF 0.0030278 TO G_small, ASK ROBIN IF THIS IS RIGHT!!!!')

    structure = []
    G_small = []
    E_small = []
    E_big = []
    G_final = []

    G_corrected_df = pd.DataFrame()

    for i in tqdm(glob.glob(outfile_location + '/*')):

        job_one = True
        file = i.split('\\')
        structure.append(file[1].split('_')[0])

        with open(i, 'r') as f:
            for line in f:
                if 'JOB NUMBER  2' in line:
                    job_one = False

                if job_one:
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        E_small_item = line.split()

                    elif 'Final Gibbs free energy' in line:
                        G_small_item = line.split()

                else:
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        E_big_item = line.split()


            G = (float(G_small_item[-2]) + 0.0030278)
            E_s = float(E_small_item[4])
            E_b = float(E_big_item[4])
            G_fin = ((E_b + G) - E_s)

            G_small.append(G)
            E_small.append(E_s)
            E_big.append(E_b)
            G_final.append(G_fin)

    G_corrected_df = pd.DataFrame({'Structure': structure, 'G_corr_au': G_small, 'E_small_au': E_small,
           'E_big_au': E_big, 'G_final_au': G_final})

    with open(outname + 'G_corrected_rawdata.csv', 'w') as p:
        print(G_corrected_df.to_csv(), file=p)


def calculate_deltaGexch(Pd_deltaG_rawdata, outname):

    '''
    Calculates donicoty (deltaG exchange) for basic nitrogens. Takes the output of calculate_Gcorrected
    as the argument and returns a dataframe of donicity values.
    :param Pd_deltaG_rawdata: output dataframe of calculate_Gcorrected()
    :param outname: output name of saved donicity dataframe
    :return:
    '''

    basehet_structures = Pd_deltaG_rawdata[Pd_deltaG_rawdata['Structure'].str.contains('basehet')].copy()

    Pd_structures = Pd_deltaG_rawdata[~Pd_deltaG_rawdata['Structure'].str.contains('basehet')]

    pyridine_G = Pd_deltaG_rawdata[Pd_deltaG_rawdata['Structure'].str.match('S49-basehet')].G_final_au.values

    py_complex_G = Pd_deltaG_rawdata[Pd_deltaG_rawdata['Structure'].str.contains('37-S49')].G_final_au.values

    exchG_H = []
    structures = []
    exchG_kcalmol = []
    exchG_rel = []

    for i in basehet_structures['Structure'].values:

        basehet_G = basehet_structures[basehet_structures['Structure'].str.endswith(i)].G_final_au.values

        het_df = Pd_structures[Pd_structures['Structure'].str.endswith(i.split('-')[0])]

        for j in het_df['Structure'].values:
            if j.split('-')[1] == i.split('-')[0]:
                if 'S49' not in j:
                    complex_G = het_df[het_df['Structure'].str.endswith(j)].G_final_au.values

                    exch_G = (py_complex_G + basehet_G) - (pyridine_G + complex_G)

                    exchG_H.extend(exch_G.tolist())
                    exchG_kcalmol.extend((exch_G * 627.5).tolist())
                    exchG_rel.extend(((exch_G * 627.5)/1.40833329762017))
                    structures.append(j)

    data = pd.DataFrame({'Structure': structures, 'ExchG_H': exchG_H, 'ExchG_kcalpermol': exchG_kcalmol,
                         'ExchG_relative': exchG_rel})


    with open(outname + '_deltaG_Energies.csv', 'w') as f:
        print(data.to_csv(), file=f)

    return Pd_structures

def assign_directed_activation_to_atom_df(atom_df, directed_activation_df):

    '''
    This function assigns the donicity values to the atom dataframe that already contains
    acidity and nucleophilicity data.
    :param atom_df: atom_df containing acidity and nucleophilicity data
    :param directed_activation_df: output dataframe of calculate_deltaGexch.
    :return: updates atom_df with assigned donicity data. writes as NCS_atom_df (see line 200)
    '''

    assigned_atom_df = pd.DataFrame()

    for id in atom_df['molecule_name'].unique():

        idx = []
        ncs = []

        mol_atom = atom_df[atom_df['molecule_name'] == id]
        mol_directed = directed_activation_df[directed_activation_df['Structure'].str.endswith(id)]

        for atom in mol_directed.Structure.values:
            idx.append(int(atom.split('-')[0]) - 34)
            ncs.append(mol_directed[mol_directed['Structure'] == atom].iloc[0]['ExchG_kcalpermol'])

        for a, b in zip(idx, ncs):
            mol_atom.loc[mol_atom['atom_index'] == a, 'shift'] = b

        assigned_atom_df = pd.concat([assigned_atom_df, mol_atom])

    with open('NCS_atom_df.csv', 'w') as f:
        print(assigned_atom_df.to_csv(), file=f)

    return assigned_atom_df
