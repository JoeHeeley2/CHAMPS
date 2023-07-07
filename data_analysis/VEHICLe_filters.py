import numpy as np
from rdkit import Chem
import pandas as pd
from VEHICLe_read import read_list_of_mol_objects
from print_file import print_to_csv
from rdkit.Chem import Draw
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import collections

def nonetype_filter(dataframe):
    for i in read_list_of_mol_objects(dataframe):
        print(i.GetNumAtoms())

def synthesis_filter(dataframe, export):

    '''
    Filters VEHICLe to give structures that have good reported syntheses.
    :param dataframe: VEHICLe subset
    :param export:
    :return:
    '''

    synthesis_filtered=dataframe[dataframe.Pgood>=0.8]

    if export == 'yes':
        print_to_csv(synthesis_filtered, 'Synthesis_filter')
    return synthesis_filtered

def beilstein_filter(dataframe, export):

    '''
    Filters structures from VEHICLe database according the beilstein index
    :param dataframe:
    :param export:
    :return:
    '''

    beilstein_filtered =dataframe[dataframe.Beilstein_hits_June_08!=0]
    if export == 'yes':
        print_to_csv(beilstein_filtered, 'Beilstein_filter')

    return beilstein_filtered

def dj1_filter(dataframe, export):

    '''
    Filter applied to VEHICLe database to get the DJ1 dataset
    :param dataframe: VEHICLe database
    :param export:
    :return:
    '''

    filtered_data = pd.DataFrame({})

    for i in dataframe:
        filtered_data[i] = i

    for j in range(len(dataframe)):
        row = dataframe.iloc[j]
        smiles = row['Smiles']
        mol = Chem.MolFromSmiles(smiles)
        type_array = np.zeros(mol.GetNumAtoms(), dtype=np.int32)

        for x, atoms in enumerate(mol.GetAtoms()):
            type_array[x] = atoms.GetAtomicNum()

        if np.count_nonzero(type_array == 6, axis=0) > 0:
            if np.count_nonzero(type_array == 7, axis=0) <= 4:
                if np.count_nonzero(type_array == 8, axis=0) == 0:
                    if np.count_nonzero(type_array == 16, axis=0) == 0:
                        filtered_data = filtered_data.append(row, ignore_index=True)

    if export == 'yes':
        print_to_csv(filtered_data, 'DJ1')

def dj2_filter(dataframe, write=True):

    '''
    Filter applied to VEHICLe database to get the expanded dataset, from which 1000 structures
    were randomly sampled to give DJ2.
    :param dataframe: VEHICLe database
    :param write: Bool, saves filteres VEHICLe subset as .csv file
    :return:
    '''

    filtered_data = pd.DataFrame({})

    for i in dataframe:
        filtered_data[i] = i

    for j in range(len(dataframe)):
        row = dataframe.iloc[j]
        smiles = row['Smiles']
        mol = Chem.MolFromSmiles(smiles)
        Hmol = Chem.AddHs(mol)
        type_array = np.zeros(Hmol.GetNumAtoms(), dtype=np.int32)

        for j, atoms in enumerate(Hmol.GetAtoms()):
            type_array[j] = atoms.GetAtomicNum()

        if np.count_nonzero(type_array == 6, axis=0) >0:
            if np.count_nonzero(type_array == 1, axis=0) >0:
                if np.count_nonzero(type_array == 7, axis=0) <=4:
                    if np.count_nonzero(type_array == 8, axis=0) <=4:
                        if np.count_nonzero(type_array == 16, axis=0) <=4:
                            filtered_data = filtered_data.append(row, ignore_index=True)

    if write:
        with open('dj2.csv', 'w') as f:
            print(filtered_data.to_csv(sep=','), file=f)

    return filtered_data

def NCN_filter(dataframe, export):
    ### This filter extracts molecules with C-C cores.

    '''
    Filter that should extract structures with pincer-like basic nitrogens connected by a C-C core.
    Not sure how well this works...
    :param dataframe: VEHICLe dataset
    :param export:
    :return:
    '''
    filtered_data = pd.DataFrame({})

    for i in dataframe:
        filtered_data[i] = i

    for j in range(len(dataframe)):
        row = dataframe.iloc[j]
        smiles = row['Smiles']
        mol = Chem.MolFromSmiles(smiles)
        type_array = np.zeros(mol.GetNumAtoms(), dtype=np.int32)

        for j, atoms in enumerate(mol.GetAtoms()):
            if atoms.GetAtomicNum() == 6:
                if len(atoms.GetNeighbors()) == 3:
                    for i in atoms.GetNeighbors():
                        if i.GetAtomicNum() == 6: ### Change condiction to == 7 for CN cores ###
                            if len(i.GetNeighbors()) == 3:
                                type_array = np.zeros(len(atoms.GetNeighbors()), dtype=np.int32)

                                for x, y in enumerate(atoms.GetNeighbors()):
                                    type_array[x] = y.GetAtomicNum()

        if np.count_nonzero(type_array == 7, axis=0) >= 2: ### Change condition to >= 3 for C-N core ###
            filtered_data = filtered_data.append(row, ignore_index=True)


    if export == 'yes':
        print_to_csv(filtered_data, 'NCN')

def directing_nitrogen_filter(vehicle_subset, ncs_atom_df, outname):


    '''
    Filter that extracts molecules that have a 5-to-6 directing mode and can engage in directed activation
    :param vehicle_subset: VEHICLe_subset
    :param ncs_atom_df: atom_df containing donicity data
    :param outname: name for saved files
    :return:
    '''

    mol_list = []
    highlight = []
    colors = {}
    legend_string = []

    for i in ncs_atom_df['molecule_name'].unique():
        smiles = vehicle_subset[vehicle_subset['Regid'] == i].Smiles.values
        mol_props = ncs_atom_df[ncs_atom_df['molecule_name'] == i]
        mol_act = directed_activation_data[directed_activation_data['Structure'].str.endswith(i)]

        acidity = mol_props[mol_props['typestr'] == 'H']
        nucleophilicity = mol_props[mol_props['typestr'] == 'C']

        if len(mol_act) != 0:
            h_idx = -404.404
            ac_c_idx = -505.505
            n_idx = int(mol_act.loc[mol_act['ExchG_relative'].idxmax()]['Structure'].split('-')[0]) - 34
            ac_drop = acidity[acidity['shift'] != 0]
            ac_idx = int(ac_drop.loc[ac_drop['shift'].idxmin()]['atom_index'])
            nuc_idx = int(nucleophilicity.loc[nucleophilicity['shift'].idxmax()]['atom_index'])

            mol = Chem.MolFromSmiles(smiles[0])
            molh = Chem.AddHs(mol)
            type_array = np.zeros(mol.GetNumAtoms(), dtype=np.int32)

            for atoms in molh.GetAtoms():
                if atoms.GetIdx() == n_idx:
                    for carbon in atoms.GetNeighbors():
                        if carbon.GetAtomicNum() == 6:
                            for hydrogen in carbon.GetNeighbors():
                                if hydrogen.GetAtomicNum() == 1:
                                    h_idx = hydrogen.GetIdx()

                elif atoms.GetIdx() == ac_idx:
                    for c in atoms.GetNeighbors():
                        if c.GetAtomicNum() == 6:
                            ac_c_idx = c.GetIdx()

            if h_idx != -404.404:
                legend_string.append(i)
                mol_list.append(mol)
                highlight.append([n_idx, ac_c_idx, nuc_idx])

    for p, j in enumerate(highlight):
        if j[1] != j[2]:
            colors[p] = {j[0]: (100, 84.3, 0), j[1]: (1,0,0), j[2]: (0,0,1)}
        else:
            colors[p] = {j[0]: (100, 84.3, 0), j[1]: (128, 0, 128)}

    img = Draw.MolsToGridImage(mol_list, molsPerRow=6, highlightAtomLists=highlight, highlightAtomColors=colors,
    legends=legend_string, subImgSize=(200, 200))

    img.save(outname + '.png', pgi=(1200))

def donicity_filter(vehicle_df, ncs_atom, outname, don_limit=3.6, nuc_limit=160, draw=True, write_class_df=True):

    '''
    classes: unreactive = 0. nucleophiliic = 1, acidic = 2, switchable=3.
    Classification based on the decision tree from CHAMPS paper and used for training/testing ML models
    :param vehicle_df: VEHICLe_subset
    :param ncs_atom: atom_df containing donicity, acidity and nucleophilicty data
    :param outname: output filename for all written files
    :param draw: Bool, if True filtered structures will be drawn and saved as outname.png
    :param write_class_df: Bool, if True Structures and their assigned classes are written to outname.csv
    :return:
    '''

    exp = ['S1072', 'S1080', 'S1863', 'S1869', 'S1873', 'S5703', 'S5704', 'S5706', 'S5707', 'S5853', 'S5854', 'S5857',
           'S6018', 'S1875', 'S291', 'S7959', 'S54', 'S8063', 'S50', 'S7972']

    ac = ['S5853', 'S5854', 'S5857', 'S6018', 'S1873', 'S5706','S1072', 'S1080']
    don = ['S6018', 'S1873']
    exp = sorted(exp)

    #ids = ncs_atom.sort_values('molecule_name')['molecule_name'].unique()
    ids = exp
    smiles = vehicle_df[vehicle_df['Regid'].isin(ids)].sort_values('Regid')['Smiles']

    mols = [Chem.AddHs(Chem.MolFromSmiles(x)) for x in smiles]

    class_ids = []
    struct_class = []
    highlight = []
    sw_mols = []
    sw_highlight = []
    sw_ids = []
    colors = {}
    ncs_atom['class'] = 0
    plot_ids = []
    dons = []
    ns = []
    acs = []
    nucs = []
    donids = []
    don_highlight = []


    for molid, mol in zip(ids, mols):
        dons.append(mol)
        donids.append(molid)
        class_ids.append(molid)
        mol_highlight = []
        #mol_donac = 1000
        directing = []
        nit_idx = {}
        added = False
        d = False
        Chem.AllChem.EmbedMolecule(mol)

        ncs_row_atom = ncs_atom[ncs_atom['molecule_name'] == molid]
        don = ncs_row_atom[ncs_row_atom['typestr'] == 'N'].sort_values('shift')
        don = don[don['shift'] != 0]
        nuc = ncs_row_atom[ncs_row_atom['typestr'] == 'C'].sort_values('shift', ascending=False).iloc[0]
        ac = ncs_row_atom[ncs_row_atom['typestr'] == 'H'].sort_values('shift')
        most_ac = ac[ac['shift'] != 0].iloc[0]


        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in don['atom_index'].values:
                shift = don.loc[don['atom_index'] == atom_idx]['shift'].values[0]

                atom.SetProp('atomNote', str(round(shift, 2)))
                for c in atom.GetNeighbors():
                    c_idx = c.GetIdx()
                    for x in c.GetNeighbors():
                        x_idx = x.GetIdx()
                        if x.GetAtomicNum() == 6:
                           for hat in x.GetNeighbors():
                               hat_idx = hat.GetIdx()
                               if hat.GetAtomicNum() == 1:
                                    ac_shift = ac['shift']
                                    #hat.SetProp('atomNote', str(round(ac_shift, 2)))
                                    dihedral = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(),
                                                                 int(atom_idx),
                                                                 int(c_idx),
                                                                 int(x_idx),
                                                                 int(hat_idx))

                                    if -1.0 < dihedral < 1.0:
                                        if hat.GetIdx() == most_ac['atom_index']:
                                            plot_ids.append(molid)
                                            ns.append(don.loc[don['atom_index'] == atom_idx]['shift'].values[0])
                                            nucs.append(nuc['shift'])
                                            acs.append(ac.loc[ac['atom_index'] == hat.GetIdx()]['shift'].values[0])
                                        directing.append([atom_idx, c_idx, x_idx, hat_idx])
                                        nit_idx[atom_idx] = shift


            elif atom_idx == nuc['atom_index']:
                if molid == 'S5857':
                    mol_highlight.append(0)
                else:
                    mol_highlight.append(int(nuc['atom_index']))

                #atom.SetProp('atomNote', str(round(nuc['shift'], 2)))

            elif atom_idx == ac.iloc[0]['atom_index']:
                for i in atom.GetNeighbors():
                    if i.GetAtomicNum() == 6:
                        mol_highlight.append(int(i.GetIdx()))

        if len(directing) > 0:
            most_donating = min(nit_idx, key=nit_idx.get)
            for i in directing:
                if i[-1] == ac.iloc[0]['atom_index']:
                    if i[0] == most_donating:
                        if nit_idx.get(i[0]) < don_limit:
                            if nuc['shift'] > nuc_limit:
                                struct_class.append(3)
                                sw_mols.append(mol)
                                sw_highlight.append(mol_highlight)
                                sw_ids.append(molid)
                                added = True
                            else:
                                struct_class.append(2)
                                added = True


                    else:
                        m = [x for x in directing if most_donating in x]
                        if m[0][-2] != nuc['atom_index']:
                            if nit_idx.get(i[0]) < don_limit:
                                struct_class.append(2)
                                added = True

                        #else:
                         #   if nit_idx.get(i[0]) < don_limit:
                          #      struct_class.append(2)
                           #     added=True
        if not added:
            if nuc['shift'] > nuc_limit:
                struct_class.append(1)

            else:
                struct_class.append(0)

        highlight.append(mol_highlight)

        don_highlight.append(mol_highlight)

    if draw:
        for p, j in enumerate(don_highlight):
            if j[0] != j[1]:
                colors[p] = {j[0]: (0, 0, 1), j[1]: (1, 0, 0)}
            else:
                colors[p] = {j[1]: (128, 0, 128)}

        img = Draw.MolsToGridImage(dons, molsPerRow=6, highlightAtomLists=don_highlight, highlightAtomColors=colors,
                                   legends=donids, subImgSize=(200, 200))

        img.save(outname + '.png', pgi=(1200))

    if write_class_df:
        for i, c in zip(class_ids, struct_class):
            ncs_atom['class'][ncs_atom.molecule_name == i] = c

        #ncs_atom.to_pickle(outname + '.pkl')

        class_df = pd.DataFrame({'Regid': class_ids, 'class': struct_class})
        class_df.to_csv(outname + '_class_df.csv')

        print('Class Distribution:', dict(collections.Counter(struct_class)))

    return plot_ids, nucs, ns, acs

def fulldata_filter(VEHICLe_dataframe, calculate_properties_dataframe, outname,
                    a_upper=1.5, a_lower=0.5, ea_upper=1.5, ea_lower=0.5, write=True):

    '''
    This function should return an rdkit image of molecules that have relative properties between the defined criteria
    :param VEHICLe_dataframe: the VEHICLe dataframe
    :param calculate_properties_dataframe: The output of the property_calculate.calculate_properties or workflow.logfile_analysis_workflow functions
    :param outname: What you would like to save the image as (recommended that
    :param a_upper:the upper bound of acidity range of interest
    :param a_lower:the lower bound of acidity range of interest
    :param ea_upper:the upper bound of nucleophilicity range of interest
    :param ea_lower:the lower bound of nucleophilicity range of interest
    :param write= Bool, if true writes image as outname.png
    :return: grid image of molecules
    '''

    if a_upper and ea_upper == 1.5:
        if a_lower and ea_lower == 0.5:
            print('Default criteria used')

    regid = list(set(calculate_properties_dataframe.Regid))
    structures=[]
    rel_a = []
    rel_ea = []

    for i in tqdm(regid):
        acidities = {}
        elec_affs = {}
        A_position = []
        Ea_position = []
        relative_acidities = []
        relative_electro_affs = []

        df = calculate_properties_dataframe.loc[calculate_properties_dataframe['Regid'] == i]
        for j in df.columns:
            if 'anion' in j:
                acidities[j] = float(df[j].values)
            elif 'bromine' in j:
                elec_affs[j] = float(df[j].values)

        for m, c in acidities.items():
            if c == (min(acidities.values())):
                relative_acidities.append(
                    24.044391262487466 / c ) if 24.044391262487466 / c  not in relative_acidities else relative_acidities
                acidic_position = list(m.split('_')[0])
                A_position.append(acidic_position)

                if len(relative_acidities) == 1:
                    break

        for f, l in elec_affs.items():
            if l == max(elec_affs.values()):
                relative_electro_affs.append(
                    l / 183.64724482984454) if l / 183.64724482984454 not in relative_electro_affs else relative_electro_affs
                nucleophilic_position = list(f.split('_')[0])
                Ea_position.append(nucleophilic_position)

        if len(relative_acidities) != len(relative_electro_affs):
            print('There is a problem with: ', i, 'please check!')

#        if (a_lower <= relative_acidities[0] < a_upper) and (ea_lower <= relative_electro_affs[0] < ea_upper):
 #           print(i, 'acidic position is:', A_position[0][0], 'with relative acidity:', relative_acidities,
  #                                        'nucleophilic position is:', Ea_position[0][0],
   #                                             'with relative electrophile affinity:', relative_electro_affs)
            structures.append(i)
            rel_a.append(relative_acidities[0])
            rel_ea.append(relative_electro_affs[0])

    mol_list = []
    legend_strings = []

    for h, k, l in zip(structures, rel_a, rel_ea):
        leg_str = 'Regid: ' + h + '\nRel Acidity: ' + str(round(k, 2)) + '\nRel Elec_Aff: ' + str(round(l, 2))
        legend_strings.append(leg_str)

        row = VEHICLe_dataframe[VEHICLe_dataframe['Regid'].str.fullmatch(h)]

        for j in row.Smiles.values:
            mol = Chem.MolFromSmiles(j)
            mol_list.append(mol)

    print('FOUND:', len(mol_list), ' STRUCTRES')
    if write:
        img = Draw.MolsToGridImage(mol_list, molsPerRow=12, legends=legend_strings, subImgSize=(200, 200))

        img.save(outname + '.png', pgi=(1200))

def djr_filter(VEHICLe_dataframe, calculate_properties_dataframe, outname,
                    djr_upper=1.2, djr_lower=0.5, write=True):

    '''
    Filter that extracts structures with DJR's between the specified limits
    :param VEHICLe_dataframe: VEHICLe_subset
    :param calculate_properties_dataframe: output dataframe of calculate_properties()
    :param outname: output filename for written files
    :param djr_upper: upper limit for djr filter
    :param djr_lower: lower limit for djr filter
    :param write: Bool, if True structures are drawn and written to outname.png
    :return:
    '''

    if djr_upper == 1.2:
        if djr_lower == 0.5:
            print('-----Default criteria used------')

    regid = list(set(calculate_properties_dataframe.Regid))
    structures = []
    positions = []
    highlight = []
    colors = {}
    selectivities = []
    rel_props = []

    for i in tqdm(regid):
        acidities = {}
        elec_affs = {}
        A_position = []
        Ea_position = []
        relative_acidities = []
        relative_electro_affs = []

        df = calculate_properties_dataframe.loc[calculate_properties_dataframe['Regid'] == i]
        for j in df.columns:
            if 'anion' in j:
                acidities[j] = float(df[j].values)
            elif 'bromine' in j:
                elec_affs[j] = float(df[j].values)

        for m, c in acidities.items():
            if c == (min(acidities.values())):
                relative_acidities.append(
                    24.044391262487466 / c) if 24.044391262487466 / c not in relative_acidities else relative_acidities
                acidic_position = list(m.split('_')[0])
                A_position.append(int(acidic_position[0]))

                if len(relative_acidities) == 1:
                    break

        for f, l in elec_affs.items():
            if l == max(elec_affs.values()):
                relative_electro_affs.append(
                    l / 183.64724482984454) if l / 183.64724482984454 not in relative_electro_affs else relative_electro_affs
                nucleophilic_position = list(f.split('_')[0])
                Ea_position.append(int(nucleophilic_position[0]))

        if len(relative_acidities) != len(relative_electro_affs):
            print('There is a problem with: ', i, 'please check!')

        djr = relative_acidities[0]/relative_electro_affs[0]

        if djr_lower < djr <= djr_upper:
            structures.append(i)
            selectivities.append(djr)
            rel_props.append([relative_acidities[0], relative_electro_affs[0]])
            positions.append(A_position + Ea_position)

    mol_list = []
    legend_strings = []

    for h, k, l, m in zip(structures, selectivities, positions, rel_props):

        leg_str = 'Regid: ' + h + ' Selectivity: ' + str(round(k, 2)) + '\nRel Acidity ' + str(round(m[0], 2)) + ' Rel Ea: ' + str(round(m[1], 2))

        legend_strings.append(leg_str)
        row = VEHICLe_dataframe[VEHICLe_dataframe['Regid'].str.fullmatch(h)]

        for j in row.Smiles.values:
            C_index_list = []
            mol = Chem.MolFromSmiles(j)
            molH = Chem.AddHs(mol)
            a = molH.GetAtoms()
            mol_list.append(mol)

            for j, atoms in enumerate(a):
                b = atoms.GetAtomicNum()
                if b == 1:
                    e = atoms.GetNeighbors()
                    for j, k in enumerate(e):
                        if k.GetAtomicNum() == 6:
                            if str(k.GetHybridization()) == 'SP2':
                                C_index = k.GetIdx()
                                C_index_list.append(C_index)

            if l[0] != l[1]:
                a_listindex = l[0] - 1
                ea_listindex = l[1] - 1
                mol_c_list = [C_index_list[a_listindex]] + [C_index_list[ea_listindex]]
                highlight.append(mol_c_list)

            else:
                a_listindex = l[0] - 1
                mol_c_list = C_index_list[a_listindex]
                highlight.append([mol_c_list])

    for p, j in enumerate(highlight):
        if len(j) > 1:
            colors[p] = {j[0]:(1,0,0), j[1]:(0,0,1)}

        else:
            colors[p] = {j[0]: (128, 0, 128)}

    if write:
        img = Draw.MolsToGridImage(mol_list, molsPerRow=6, highlightAtomLists=highlight, highlightAtomColors=colors,
        legends=legend_strings, subImgSize=(200, 200))

        img.save(outname + '.png', pgi=(1200))

def atom_filter(vehicle_df, relative_properties_dataframe, criteria, outname, write_df=True, write_pic=True):
    '''
    Filters vehicle structures according to criteria
    :param vehicle_df: VEHILCe subset
    :param relative_properties_dataframe: output of calc_relative_properties()
    :param criteria: shoud be a dictionary of atomic number and count you want in the target structure
    :param write_df: Bool, if True datafram is written as outname.csv
    :param write_pic:Bool, if true, structures are drawn and written to outname.png
    :return:
    '''

    regid = vehicle_df['Regid']

    filtered_data = pd.DataFrame()
    structures = []
    positions = []
    highlight = []
    colors = {}
    selectivities = []
    rel_props = []
    mol_list = []
    legend_strings = []

    for i in tqdm(regid):

        truth_index = []
        row = vehicle_df[vehicle_df['Regid'] == i]
        rel_row = relative_properties_dataframe[relative_properties_dataframe['Regid'] == i]
        smiles = row['Smiles'].values[0]
        mol = Chem.MolFromSmiles(smiles)
        Hmol = Chem.AddHs(mol)
        type_array = np.zeros(Hmol.GetNumAtoms(), dtype=np.int32)

        for j, atoms in enumerate(Hmol.GetAtoms()):
            type_array[j] = atoms.GetAtomicNum()

        for atom, count in criteria.items():
            if np.count_nonzero(type_array == atom, axis=0) <= count:
                truth_index.append(True)

        if len(truth_index) == len(criteria.keys()):
            filtered_data = filtered_data.append(row)

    return filtered_data

def dataset_count(vehicle_df):
    '''
    Counts the number of CH bonds in a given dataset
    :param vehicle_df: the vehicle df you want ot count
    :return:
    '''

    smiles = vehicle_df['Smiles']

    for i in tqdm(smiles):
        mol = Chem.MolFromSmiles(i)
        Hmol = Chem.AddHs(mol)
        type_array = np.zeros(Hmol.GetNumAtoms(), dtype=np.int32)

        for j, atoms in enumerate(Hmol.GetAtoms()):
            type_array[j] = atoms.GetAtomicNum()

        for atom, count in criteria.items():
            if np.count_nonzero(type_array == atom, axis=0) <= count:
                truth_index.append(True)

def remove_NHs(vehicle_subset, atom_df, pair_df):

    '''
    Filter that removes structures containing NH bonds from dataset
    :param vehicle_subset: VEHICLe_subset
    :param atom_df: atom_df to filter
    :param pair_df: pair_df to filter
    :return:
    '''

    filtered = []
    smiles = vehicle_subset['Smiles'].values
    regids = vehicle_subset['Regid'].values

    mols = [Chem.MolFromSmiles(x) for x in smiles]

    for mol, regid, smiles in zip(mols, regids, smiles):
        add = True
        Hmol = Chem.AddHs(mol)

        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                for h in atom.GetNeighbors():
                    if h.GetAtomicNum() == 1:
                        add = False

        if add:
            filtered.append(regid)

    filtered_atom = atom_df[atom_df['molecule_name'].isin(filtered)]
    filtered_pair = pair_df[pair_df['molecule_name'].isin(filtered)]

    filtered_atom.to_csv('DJ1_full_NoNHs_atom.csv')
    filtered_pair.to_csv('DJ1_full_NoNHs_pair.csv')

def benzothiophene(dataframe, export):
    '''
    apply this to the VEHICLe dataframe to get benzothiophene. For Thesis only really.
    '''

    filtered_data = pd.DataFrame({})

    for i in dataframe:
        filtered_data[i] = i

    for j in range(len(dataframe)):
        row = dataframe.iloc[j]
        smiles = row['Smiles']

        if 'n' not in smiles:
            if 'N' not in smiles:
                if 'o' not in smiles:
                    if 'O' not in smiles:
                        print(row['Regid'], row['Smiles'])




