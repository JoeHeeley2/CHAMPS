from datetime import date
import glob

def type_dict(warning=True):

    type_dict = {1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl',
                 35: 'Br', 46: 'Pd', 53: 'I'}

    return type_dict

def xyzcoords_from_type_array(filename, zipped_type_array, warning=True):

    Atoms = type_dict(warning=warning)
    with open(filename, 'w') as f:
        for i in zipped_type_array:
            for atomicnum, atom in Atoms.items():
                if i[0] == atomicnum:
                    print(atom, i[1], '\t', i[2][0], '\t', i[2][1], '\t', i[2][2], file=f)

def xyz_from_com(comfilename, save_loc, warning=True):

    '''
    Writes a .xyz file from gaussian .com file.
    :param comfilename: name of comfile to convert
    :param save_loc: output directory
    :param warning: Bool, flags up atom types in type_array if True. Be aware that if you are using atom
                    types that are not in the type array they will be missed from the output file.
    :return:
    '''

    split = comfilename.split('\\')
    name = split[-1].split('.')
    atoms = type_dict(warning=warning).values()
    with open(comfilename) as p:
        xyz = []
        read = p.readlines()[5:]
        remove_command=read[1:] ### Why did I do this? ###
        charge_multip= read[0]
        charge = int(charge_multip[0] + charge_multip[1])

        for line in remove_command:
            if not line.strip():
                continue

            elif 'F' in line and 'B' in line:
                pass

            else:
                for i in atoms:
                    if i in line:
                        xyz.append(line.strip('\n'))
        numatoms= len(xyz)
        xyzfilename = save_loc + '/' + name[0] + '.xyz'

        with open(xyzfilename, 'w') as g:
            print(numatoms,'\n', 'charge=' + str(charge) + '=', file=g, sep='')

            for x in xyz:
                line=str(x)
                print(line.strip('\n'), file=g)

    return xyzfilename

def write_orca_file_for_directed_activation_opt_and_freq(regid, indexfilename, inpfilename):

    '''
    Writes an orca inputfile for the geometry optimisation and frequency calculation for
    calculation of donicity. Defaults to wB97x-D3BJ/def2-TZVP with def2-SVP on Pd for geom opt
    and B97x-D3BJ/def2-TZVP on all atoms for freq calc. Both jobs are written to the same input file.

    Required the output of nitrogen_palladation.palladate_nitrogens() function.

    :param regid: VEHICLe Regid of structure
    :param indexfilename: Index filename of structure
    :param inpfilename: desired Orca input filename
    :return:
    '''

    xyzfilename = inpfilename.split('/')[-1].split('.')[0] + '.xyz'

    with open(inpfilename, 'w') as f:

        f.write('! wB97x-d3bj def2-TZVP CPCM(Toluene) xyzfile opt freq\n')
        f.write('%base "' + regid + 'OptFreq"\n')
        f.write('%maxcore 2000\n')
        f.write('%scf MaxIter 1500 end\n')
        f.write('%pal nprocs 10\n')
        f.write('end\n')
        f.write('\n')
        f.write('*xyz 1 1\n')

        with open(indexfilename) as p:

            for indexline in p:
                if 'Pd' in indexline:
                    print(indexline[:2].strip('\n'), indexline[4:].strip('\n').lstrip(), 'newgto "def2-SVP" end', file=f)
                else:
                    print(indexline[0].strip('\n'), indexline[4:].strip('\n').lstrip(), file=f)

        f.write('*\n')
        f.write('\n')
        f.write('$new_job\n')
        f.write('! wB97x-d3bj def2-TZVP CPCM(Toluene) printbasis\n')
        f.write('%base"' + regid + 'SinglePoint"\n')
        f.write('\n')
        f.write('*xyzfile 1 1 ' + regid + 'OptFreq.xyz')
        f.write('\n')

def frequency_calc_setup(list_of_regids):

    '''
    Writes gaussian input file for single point frequency calculation from optimised structures by reading
    gaussian .log files. Check filelocations on line 119 and 133 to ensure you are reading/saving in the
    right place.
    :param list_of_regids:  list of structures for calculation setup.
    :return:
    '''

    types = type_dict()

    for i in list_of_regids:

        logfiles = glob.glob('dj1/logfiles/*' + i + '*')

        exc = [x for x in logfiles if all(name not in x for name in ['Fluorinated',  'Methylated'])]

        amols = file_to_aemol(logfiles=exc)

        for file, mol in zip(exc, amols):

            filesplit = file.split('\\')[-1].split('_')

            filename = filesplit[0] + '_' + filesplit[1] + '_FREQ_FROM_OPT_COORDS'

            atom_xyz = zip(mol.structure['types'], list(mol.structure['xyz']))

            with open('dj1/frequency_for_publication/' + filename + '.com', 'w') as p:

                p.write('%chk=' + filename + '.chk')
                p.write('\n')

                p.write('# freq wb97xd/6-31g(d)')

                p.write('\n')
                p.write('\n')
                p.write(filesplit[0] + '_' + filesplit[1])
                p.write('\n')
                p.write('\n')

                if 'anion' in filesplit[1]:
                    p.write('-1 1\n')
                elif 'bromine' in filesplit[1]:
                    p.write('1 1\n')
                else:
                    p.write('0 1\n')

                for b, c in atom_xyz:
                    print(types[b], *c.tolist(), file=p)

                p.write('\n')

