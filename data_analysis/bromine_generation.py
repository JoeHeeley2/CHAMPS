from datetime import date
import glob
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob


def bromines_from_smiles(dataframe, workflow=False):
    '''
        Function writes gaussian input files for the calculations of relative electrophile affinities
        of aromatic heterocycles starting from the SMILES string. It brominates each sp2 hybridised CH bond
        and writes a com file for the corresponding Wheland intermediate. Requires a directory of 'txt_files'
        and 'comfiles'. If the Wheland intermediate is generated next to a heteroatom-heteroatom bond,
        the heteroatom-heteroatom bond is frozen to prevet it from dissociating.
        Because of this, all sturctures are pre-optimised using molecular mechanics.

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
        C_index_list = []
        C_numbering = []
        Het_index = []
        comfiles = glob.glob('comfiles/' + str(l) + '_*')
        bromine_comfiles = [x for x in comfiles if 'anion' not in x]
        neutral_comfiles = [y for y in bromine_comfiles if 'bromine' not in y]
        index_file = glob.glob('txt_files/' + str(l) + '_indexing.txt')
        row = dataframe[dataframe.Regid == l]
        smiles = row.Smiles.item()

        mol = Chem.MolFromSmiles(smiles)
        molH = Chem.AddHs(mol)
        molc = AllChem.Compute2DCoords(molH)
        a = molH.GetAtoms()

        for j, atoms in enumerate(a):
            b = atoms.GetAtomicNum()
            d = atoms.GetIdx()
            if b == 1:
                e = atoms.GetNeighbors()
                for j, k in enumerate(e):
                    if k.GetAtomicNum() == 6:
                        if str(k.GetHybridization()) == 'SP2':
                            C_index = k.GetIdx()
                            C_index_list.append(C_index)
                            for w in k.GetNeighbors():
                                if w.GetAtomicNum() != 6:
                                    for x in w.GetNeighbors():
                                        if x.GetAtomicNum() != 6:
                                            if x.GetAtomicNum() != 1:
                                                    hets = [w.GetIdx(), x.GetIdx(), C_index]
                                                    Het_index.append(hets)

        for x in range(len(C_index_list)):
            C_numbering.append(x + 1)

        match_Cnum_Cindex = zip(C_index_list, C_numbering)

        for file in neutral_comfiles:
            todays_date = str(date.today())
            split = file.split('\\')
            name = split[1].split('_')
            for m in index_file:
                for n, o in match_Cnum_Cindex:
                    with open(m) as p:
                        for line in p:
                            comindex = line[2] + line[3]
                            if float(n) == float(comindex):
                                bromine_comfilename = 'comfiles/' + name[0] + '_' + str(o) + 'bromine_' + todays_date + '_' + name[2] + '-' + name[3] + '_opt.com'
                                bromine_chkfilename = name[0] + '_' + str(o) + 'bromine_' + todays_date + '_' + name[
                                    2] + '-' + name[3] + '_opt.chk'
                                split = line.split(' ')
                                Br_xcoord = float(split[3])
                                Br_ycoord = float(split[5])
                                Br_zcoord = float(split[7])

                                Br_coords = (str(Br_xcoord) + ' ' + str(Br_ycoord) + ' ' + str(Br_zcoord))

                        with open(bromine_comfilename, 'w') as q:
                            with open(file) as f:
                                for comline in f:
                                    if comline.startswith('%chk='):
                                        print('%chk=' + bromine_chkfilename, file=q)

                                    elif comline.startswith('# opt'):
                                        print('# opt=modredundant wb97xd/6-31g(d)\n', file=q)

                                    elif comline == '0 1\n':
                                        print('\n1 1', file=q)

                                    elif str(Br_xcoord) in comline:
                                        comsplit = comline.split()
                                        if Br_xcoord == float(comsplit[1]):
                                            if Br_ycoord == float(comsplit[2]):
                                                if Br_zcoord == float(comsplit[3]):
                                                    print(comline.strip('\n'), file=q)
                                                    print('Br', Br_xcoord, Br_ycoord, str((1.870000000000000000 + Br_zcoord)), file=q)

                                    elif comline == '\n':
                                        pass

                                    else:
                                        print(comline.strip('\n'), file=q)

                            print('\nB ' + str(int(float(n+1))) + ' '
                                          + str(int(float(n+2))), ' F', file=q)

                            if len(Het_index) > 0:
                                for h, k, v in Het_index:
                                    if v == n:
                                        if h > n:
                                            atom1 = h+2
                                        else:
                                            atom1 = h+1

                                        if k > n:
                                            atom2 = k + 2
                                        else:
                                            atom2 = k + 1

                                        print('B ' + str(int(float(atom1))) + ' '
                                              + str(int(float(atom2))), ' F', file=q)
