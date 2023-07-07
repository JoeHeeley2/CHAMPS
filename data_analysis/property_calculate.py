import pandas as pd


def calculate_properties(dataset, outname, write=True):

    '''
    This function take the raw data from logfile_read.energy_readdata() and calculates acidity and
    nucleophilicity. The resultant calculate_properties dataframe is used as inuts for many of the plots
    and ML input workflows.
    :param dataset: raw data dataframe from energy_readdata()
    :param outname: output filename for any written files
    :param write: Bool, if True then data is written as a .csv file
    :return:
    '''
    regId = list(set(dataset.Regid))
    property_data=pd.DataFrame(columns=['Regid'])

    for id in regId:
        regid_dict= {'Regid': id}
        regid_data = dataset[dataset['Regid'].str.fullmatch(id)]
        alist = list(regid_data.itertuples(index=False))
        basehet = (regid_data[regid_data['Structure'].str.contains('Base_het')]).Energy

        for j in basehet:
            for i in alist:
                if 'anion' in i.Structure:
                    acidity_au =(i[3] + -229.010318493) - (j + -228.403959536)
                    acidity_kcalpermol = acidity_au * 627.5
                    regid_dict[i[2] + '_acidity_kcalpermol'] = acidity_kcalpermol

                elif 'bromine' in i.Structure:
                    elec_affin_au = (j + -2571.19136375) - i[3]
                    elec_affin_kcalpermol = elec_affin_au * 627.5
                    regid_dict[i[2] + '_electrophile_affinity_kcalpermol'] = elec_affin_kcalpermol

        property_data=property_data.append(regid_dict, ignore_index=True)

    if write:
        with open(outname, 'w') as f:
            print(property_data.to_csv(sep=','), file=f)
    return property_data

def calculate_deltaG(deltaG_rawdata, outname, write=True):


    '''
    Function to calculate delta G from the raw data. The raw data must be the output from the deltaG_readdata
    function. Frequency calcs are necessary to calculate this!
    :param deltaG_rawdata: output dataframe from the deltaG_readdata function
    :param outname: output filename for the written files
    :param write: Bool, if True the data frame is written as a .csv file
    :return:
    '''

    regId = list(set(deltaG_rawdata.Regid))
    property_data = pd.DataFrame(columns=['Regid'])

    for id in regId:
        regid_dict = {'Regid': id}
        regid_data = deltaG_rawdata[deltaG_rawdata['Regid'].str.fullmatch(id)]
        alist = list(regid_data.itertuples(index=False))
        basehet = (regid_data[regid_data['Structure'].str.contains('Base_het')]).Energy

        for j in basehet:
            for i in alist:
                if 'anion' in i.Structure:
                    acidity_au = (i[-1] + -229.046860464999) - (j + -228.477217341)
                    acidity_kcalpermol = acidity_au * 627.5
                    regid_dict[i[2] + '_acidity_kcalpermol'] = acidity_kcalpermol

                elif 'bromine' in i.Structure:
                    elec_affin_au = (j + -2571.16789474) - i[-1]
                    elec_affin_kcalpermol = elec_affin_au * 627.5
                    regid_dict[i[2] + '_electrophile_affinity_kcalpermol'] = elec_affin_kcalpermol

        property_data = pd.concat([property_data, pd.DataFrame(regid_dict, index=[0])], ignore_index=True)

    if write:
        with open(outname + '.csv', 'w') as f:
            print(property_data.to_csv(sep=','), file=f)
    return property_data

def calculate_relative_properties(calculate_properties_dataframe, outname, write=True):


    '''
    Calculates acidity and nucleophilicity values relative to pyrazolo[1,5-a]pyrimidine.
    :param calculate_properties_dataframe: output dataframe of calculate_properties()
    :param outname: filename for all output files
    :param write: Bool, if True relative_properties_dataframe is saved as outname.csv
    :return:
    '''

    regid=list(set(calculate_properties_dataframe.Regid))
    rel_props_df = pd.DataFrame([])

    for i in regid:
        rel_props = {}
        rel_props['Regid'] = i
        df = calculate_properties_dataframe.loc[calculate_properties_dataframe['Regid'] == i]

        for j in df.columns:
            if 'anion' in j:
                acidity = float(df[j].values)
                rel_props['rel_' + j] = (24.044391262487466/acidity)

            elif 'bromine' in j:
                elec_aff = float(df[j].values)
                rel_props['rel_' + j] = (elec_aff/183.647244829844)
        rel_props_df = rel_props_df.append(rel_props, ignore_index=True)

    fill_nan = rel_props_df.fillna(0)

    if write:
        with open(outname, 'w') as f:
            print(fill_nan.to_csv(sep=','), file=f)

    return fill_nan


def find_average_diff(calculate_properties_dataframe):

    '''
    Function that finds the avaerage different between top two acidic/nucleophilic sites across the whole
    dataset. Half of this value is the limit for machine learning predictions to be useful
    :param calculate_properties_dataframe: output dataframe from calculate_properties()
    :return:
    '''

    exp = ['S1072', 'S1080', 'S1863', 'S1869', 'S1873', 'S5703', 'S5704', 'S5706', 'S5707', 'S5853', 'S5854', 'S5857',
           'S6018', 'S1875', 'S291', 'S8063', 'S54', 'S7972']
    regids = calculate_properties_dataframe['Regid']
    diff_acidity = []
    diff_elec = []

    for i in exp:
        acidity = []
        elec_aff = []
        props = calculate_properties_dataframe[calculate_properties_dataframe['Regid'] == i]
        prop_nan = props.fillna(0)

        anions = [j for j in prop_nan.columns if 'anion' in j]
        bromines = [k for k in prop_nan.columns if 'bromine' in k]

        for l in anions:
            acidity = acidity + prop_nan[l].values.tolist()

        for m in bromines:
            elec_aff = elec_aff + prop_nan[m].values.tolist()

        if len(acidity) > 0:
            a_max = max(acidity)
            a2_max = max([x for x in acidity if x != a_max])

            if a2_max != 0:
                diff_acidity.append(a_max - a2_max)

        if len(elec_aff) > 0:
            b_max = max(elec_aff)
            b2_max = max([y for y in elec_aff if y != b_max])


            if b2_max != 0:
                diff_elec.append(b_max - b2_max)

    print('Average delta acidity: ', sum(diff_acidity) / len(diff_acidity))
    print('Average delta Electrophile Affinity: ', sum(diff_elec) / len(diff_elec))

def find_average_diff_ncs(ncs_atom_df):


    '''
    Finding average difference of top two donating nitrogens in the dataset. Basically the same function
    as above but for donicity.
    :param ncs_atom_df: atom_df containing donicity data
    :return:
    '''

    diffs = []

    for regid in ncs_atom_df['molecule_name'].unique():

        row = ncs_atom_df[ncs_atom_df['molecule_name'] == regid]
        shifts = row[row['typestr'] == 'N']['shift'].values

        non_zero = [x for x in shifts if x !=0]

        if len(non_zero) > 1:
            max_ncs = max(non_zero)
            if len(non_zero) == 2:
                max2_ncs = [x for x in list(non_zero) if x != max_ncs]
            else:
                max2_ncs = max([x for x in non_zero if x != max_ncs])

            diffs.append((max_ncs - max2_ncs))

    print('Average delta donicity: ', sum(diffs) / len(diffs))