import glob
import pandas as pd
from neutral_generation import basehet_from_smiles
from anion_generation import anions_from_smiles
from bromine_generation import bromines_from_smiles
from logfile_read import energy_readdata
from calculation_check import opt_check, output_structure_check
from property_calculate import calculate_properties
from property_calculate import calculate_relative_properties
from plots import plot_activation_map
from impression_input import file_to_aemol, write_imp_input
from sampling import get_tm_df, get_fps_ids
from calculation_check import comfile_check
from tqdm import tqdm
import os

def comfile_generation_workflow(dataframe):

    '''
    Wrapper for functions that generate Gaussian 16 inputs for calculation of acidities and nucleophilicities.
    Comfiles will be written to the 'comfiles' directory.
    :param dataframe: VEHICLe dataframe containing only structures you want to generate comfiles for.
    :return:
    '''
    basehet_from_smiles(dataframe, workflow=True)
    anions_from_smiles(dataframe, workflow=True)
    bromines_from_smiles(dataframe, workflow=True)

    comfiles = glob.glob('comfiles/*')
    regids = dataframe['Regid']

    for i in comfiles:
        if not os.path.isdir(i):
            with open(i, 'a') as f:
                f.write('\n\n')
                f.close()

    #comfile_check(comfiles=comfiles, sampled_structures=regids)

def logfile_analysis_workflow(logfiles, comfiles, outname, calc_check=True, calc_data=True, calc_rel_data=True,
                              plot_map=False, imp_input=True, write=True):

    '''
    Wrapper for functions that analyse gaussian .log files and calculate acidity and nucleophilicity.
    Logfiles that have failed or give a high varience from the input structure are flagged before
    calculating properties. It is possible to generate relative data and plot the activaiton map using
    keywords (see below).

    You must ensure that each logfile has a corresponding comfile if calc_check = True, else it will throw
    an error.

    :param logfiles: list of logfiles with completed calculations
    :param comfiles: list of comfiles that were inputs for the above logfiles
    :param outname: filenames for all generated files.
    :param calc_check: Bool, if True then optimised structures will be read form the logfiels
                        and compared to the input.
    :param calc_data: Bool, if True the calculate_properties() function is called to calculate the acidities
                        and nucleophilicities
    :param calc_rel_data: Bool, if True the calculate_relative_properties() function is called to defined
                            acidities and nucleophilicities relative to pyrazolo[1,5-a]pyrimidine
    :param plot_map: Bool, if True activation map will be plotted and written as outfile.png
    :param imp_input: Bool, actually does nothing yet, but will write ML inputs once I fix it
    :param write: Bool, If True all generated dataframes will be saved as .csv files
    :return:
    '''

    if calc_check:
        failed_runs = []
        for i in tqdm(logfiles):
            if opt_check(i, 'Normal termination of Gaussian 16'):
                failed_runs.append(i)

        bad_structures = output_structure_check(comfiles=comfiles, logfiles=logfiles)
        print('Found', len(failed_runs), 'failed structures:', failed_runs)

        with open(outname + '_failed_structures.txt', 'w') as f:
            for i in failed_runs:
                print(i, file=f)

        with open(outname + '_bad_structures.txt', 'w') as p:
            for j in [x for x in bad_structures if x not in failed_runs]:
                print(j, file=p)

    if calc_data:
        read_data = energy_readdata(logfiles, outname=(outname + '_rawdata.csv'), write=write)
        calculate_properties(pd.read_csv(outname + '_rawdata.csv'), outname=(outname + '_fulldata.csv'), write=write)

        if calc_rel_data:
            calculate_properties_df = pd.read_csv(outname + '_fulldata.csv')
            calculate_relative_properties(calculate_properties_dataframe=calculate_properties_df,
                                          outname= outname + '_fulldata_relative.csv', write=True)

    if plot_map:
        plot_activation_map(pd.read_csv(outname + '_fulldata.csv'), outname + '_activation_map.png')


def impression_input_workflow(neutral_logfiles, fulldata_df, outname, write=True):
    '''
    Wrapper for all functions needed to generate the atom and pair dataframes for the training and
    testing of machine learning models.

    :param logfiles: should be the neutral logfiles for the structures you want to use
    :param fulldata_df: the output of calculate_properties or logfile_analysis_workflow
    :param outname: What you want to name the output atom and pair dataframes
    :param write: Bool, will write the dataframes to .csv files if True
    :return: atom_df, pair_df for structures inputted structures
    '''

    aemols = file_to_aemol(neutral_logfiles, write=False)
    imp_input = write_imp_input(aemols=aemols, fulldata_df=fulldata_df, outname=outname, write=write)

    return imp_input[0], imp_input[1]


def logfile_fingerprint_sampling_workflow(neutral_logfiles, training_structures, vehicle_df, outname,
                                  fulldata_df, num=100, imp_input=True, write=True):
    '''
    Wrapper for functions relating to tanimoto similarity coefficients and diversity sampling.

    :param neutral_logfiles: list of logfiles that you want to sample
    :param training_structures: vehicle regid's of the structures you want to sample from
    :param vehicle_df: the full vehicledatabase as a dataframe
    :param outname: what you want to save your files as
    :param fulldata_df: the output of the properties_calculate.calculate_properties function
    :param num: the number of structures you want to sample
    :param imp_input: boolean, writes the input atom and pari dataframes for IMPRESSION Gen I if True
    :param write: boolean, saves all generated files if True

    :return: a list of regids that correspond to the sampled structures
    '''

    tm_df_array = get_tm_df(training_structures=training_structures, vehicle_df=vehicle_df)
    fpsample = get_fps_ids(tm_df=tm_df_array[0], tm_array=tm_df_array[1], num=num, outname=outname, write=write)

    sample_structures = []
    for i in neutral_logfiles:
        name = i.split('\\')[1].split('_')[0]
        if name in fpsample:
            sample_structures.append(i)

    if imp_input:
        impression_input_workflow(logfiles=sample_structures, fulldata_df=fulldata_df, outname=outname + str(num),
                                    write=write)

    return fpsample


def vehicle_fingerprint_sampling_workflow(vehicle_dataframe, outname, num=100, write=True):

    '''
    Wrapper for functions relating to Tanimoto similarity and diversity sampling starting
    from SMILES strings provided as part of the VEHICLe database.

    :param vehicle_dataframe: VEHICLe_subset. This wil provide the sample pool for diversity sampling.
    :param outname: output name for any files written
    :param num: Number of structures you want to sample. Default: 100
    :param write: Bool, if True all dataframes are written as .csv files
    :return: sample_dataframe, dataframe with sampled structures
    '''

    sampled_dataframe = pd.DataFrame([])
    regids = vehicle_dataframe['Regid']
    tm_df_array = get_tm_df(training_structures=regids, vehicle_df=vehicle_dataframe)
    fpsample = get_fps_ids(tm_df=tm_df_array[0], tm_array=tm_df_array[1], num=num, outname=outname, write=write)

    for i in fpsample:
        row = vehicle_dataframe[vehicle_dataframe.Regid == i]
        sampled_dataframe = sampled_dataframe.append(row)

    if write:
        with open(outname + str(num) +'.csv', 'w') as f:
            print(sampled_dataframe.to_csv(sep=','), file=f)

    return sampled_dataframe
