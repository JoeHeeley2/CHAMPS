import pandas as pd
import glob
from neutral_generation import write_gaussian_command_with_regid
from file_generation import type_dict
from mol_translator import aemol
import numpy as np

def opt_check(file, string_to_search):

    '''
    Checks file to see if it contains string_to_search
    :param file: file to check
    :param string_to_search: string to search
    :return:
    '''
    with open(file, 'r') as f:
        for line in f:
            if string_to_search in line:
                return True
    return False

def g16_scfread(file):

    '''
    Check gaussian16 .log file for final SCF energy
    :param file: file to read
    :return:
    '''

    energy = -404
    with open(file, 'r') as f:
        for line in f:
            if 'SCF Done' in line:
                items = line.split()
                energy = float(items[4])

    return energy

def g16_TCGread(file):

    '''
    Reads gaussian16 .log files and extracts thermal correction to Gibbs Free Energy
    :param file: file to check
    :return:
    '''
    TCG = -404

    with open(file, 'r') as f:
        for line in f:
            if 'Thermal correction to Gibbs Free Energy' in line:
                items2 = line.split()
                TCG = float(items2[-1])
    return TCG

def orca_scfread(file):

    '''
    reads Orca .out files and extracts final SCF energy
    :param file: .out file to read
    :return:
    '''
    energy = -404
    with open(file, 'r') as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                items = line.split()
                energy = float(items[-1])

    return energy

def orcaxyz_to_comfile(file):

    '''
    Writes and gaussian .com file from an optimised Orca .xyz files
    :param file: orca output .xyz fie containing optimised structure
    :return:
    '''

    structure = file.split('.')[0]

    comfilename = structure + '.gjf'
    write_gaussian_command_with_regid(structure, 'tmp.chk')

    with open(comfilename, 'w') as fn:
        with open(file, 'r') as f:
            lines = f.readlines()[2:]
            with open('Gaussian_command.txt') as fp:

                    for gaussline in fp:
                        print(gaussline.strip('\n'), file=fn)

                    for xyzline in lines:
                        print(xyzline.strip('\n'), file=fn)

def energy_readdata(logfiles, outname, write=True):

    '''
    Reads SCF energies from list of logfiles and writes them into a dataframe. gets all other information
    from the file title, which is typically in the form:
                                'REGID_STRUCTURE-TYPE_DATE_FUNCTIONAL-BASISSET_CALCTYPE.com
    :param logfiles: list of logfiles to read
    :param outname: output filename for raw data file
    :param write: Bool, if True this raw data is written to a .csv file
    :return:
    '''

    energies = []
    date=[]
    structure=[]
    RegId = []
    theory=[]
    calc=[]

    for file in logfiles:
        energy = g16_scfread(file)
        if energy != -404:
            energies.append(energy)
            x = file.split('\\')
            z = x[-1].split('_')
            RegId.append(z[0])
            theory.append(z[3])
            calc.append(z[-1])

            if 'anion' in z[1]:
                structure.append(z[1])
                date.append(z[2])
            elif 'bromine' in z[1]:
                structure.append(z[1])
                date.append(z[2])

            else:
                structure.append('Base_het')
                date.append(z[1])

        else:
            print(file, 'Energy not found')

    dict = {'Regid': RegId, 'Structure': structure, 'Energy': energies, 'Date': date, 'Functional/basisset': theory, 'Calculation': calc}
    data = pd.DataFrame(dict)

    if write:
        with open(outname, 'w') as f:
            print(data.to_csv(sep=','), file=f)

    return data

def check_freq_calcs(freq_logfile_location):

    '''
    Checks for failed runs and imaginary frequencies in gaussian16 .log files for frequency calculations
    :param freq_logfile_location: location of all frequency calculation .log files
    :return:
    '''

    files = glob.glob(freq_logfile_location + '/*.log')
    failed_runs = []
    imaginary_freqs= []
    imaginary_files = []

    for i in files:
        freq = []
        if opt_check(i, 'Error'):
            failed_runs.append(i)

        with open(i, 'r') as f:
            for line in f:
                if 'Frequencies' in line:
                    for item in [-1, -2, -3]:
                        freq.append(float(line.split()[item]))

        negatives = [x for x in freq if x < 0]

        if len(negatives) != 0:
            imaginary_freqs.append(negatives)
            print(i)
            imaginary_files.append(i)

    print('Found', len(failed_runs), 'Failed runs')
    print('Found', len(imaginary_freqs), 'Imaginary Frequencies')

def deltaG_readdata(freq_logfile_location, outname):

    '''
    Reads gaussian16 logfiles and extracts SCF energies and thermal correction to gibbs free energies.
    Also calculates the gibbs free energy by taking the sum of the SCF energy and thermal correction
    to gibbs free energy.
    :param freq_logfile_location: locaation of .log files that contain frequency calculation data
    :param outname: output filename for all written files
    :return:
    '''

    check_freq_calcs(freq_logfile_location)

    energies = []
    TCG_list = []
    Regid = []
    deltaG_au = []
    structure = []
    calctype = []

    for i in glob.glob(freq_logfile_location + '/*.log'):
        energy = g16_scfread(i)
        TCG = g16_TCGread(i)
        if energy != -404:
            energies.append(energy)
            x = i.split('\\')
            z = x[-1].split('_')
            Regid.append(z[0])
            calctype.append(z[2])

            if 'anion' in z[1]:
                structure.append(z[1])

            elif 'bromine' in z[1]:
                structure.append(z[1])

            else:
                structure.append('Base_het')

        if TCG != -404:
            TCG_list.append(TCG)

            deltaG_au.append(energy + TCG)

        else:
            print(i, 'Energy not found')

    dict = {'Regid': Regid, 'Structure': structure, 'Energy': energies, 'TCG': TCG_list, 'Calculation': calctype,
             'DeltaG_au': deltaG_au}

    data = pd.DataFrame(dict)


    with open(outname + '.csv', 'w') as y:
        print(data.to_csv(sep=','), file=y)

def comfile_to_aemol(comfile):

    '''
    Writes AutoEnrich molecule object from gaussian16 .com file. This is only really used to make sure
    that the inputfiles containg the correct structure.
    :param comfile: .com file to convert
    :return:
    '''


    file = open(comfile)
    typ_dict = type_dict()

    lines = file.readlines()

    id = lines[3]

    types = []
    xyz = []

    coords = lines[6:]

    for i in [x for x in coords if x != '\n' and 'F\n' not in x]:

        linsplit = i.split(' ')

        types.extend([mr for mr, symb in typ_dict.items() if symb == linsplit[0]])

        remove_space = [x for x in linsplit[1:] if not x.isspace()]

        xyz.append([float(y) for y in remove_space if len(y) > 2])


    #for index, atom in enumerate(len(types)):

    amol = aemol(molid=id)

    amol.structure['size'] = len(types)
    amol.structure['xyz'] = np.array(xyz)
    amol.structure['types'] = np.array(types)

    return amol


