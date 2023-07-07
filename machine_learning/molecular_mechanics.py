from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from mol_translator import aemol
from model.features.dscribe_inputs import add_local_many_body_tensor_to_atom_df
from mol_translator.imp_converter import dataframe_write as dfw
from model.dscribe_model import DSCRIBEmodel
from model.classifier_model import ClassifierModel
import datetime

def write_mm_structures(vehicle_df):

    '''
    This functions performs a molecular mechanics optimisation of the experimental structures
    tested in the CHAMPS paper

    :param vehicle_df: VEHICLe database containing SMILES strings of experimetnal structures.

    :return: Atom Dataframe containing experimental structures optimised by molecular mechanics
            AutoEnrich Molecule objects of optimised structures
    '''

    exp = ['S1072', 'S1080', 'S1863', 'S1869', 'S1873', 'S5703', 'S5704', 'S5706', 'S5707', 'S5853', 'S5854', 'S5857',
           'S6018', 'S1875', 'S291', 'S8063', 'S54', 'S7972']

    exp_df = vehicle_df[vehicle_df['Regid'].isin(exp)]
    #exp_df = vehicle_df

    mols = [Chem.MolFromSmiles(x) for x in exp_df.Smiles.values]
    ids = exp_df['Regid']
    amols = []
    print('Starting MM Opt:', datetime.datetime.now())
    for i, m in zip(ids, mols):

        molH = Chem.AddHs(m)
        Chem.AllChem.EmbedMolecule(molH)

        AllChem.MMFFOptimizeMolecule(molH)

        amol = aemol(molid=i)
        amol.from_rdkit(molH)
        amols.append(amol)
    print('Finished MM Opt:', datetime.datetime.now())
    atom_df = dfw.make_atom_df(amols)
    #pair_df = dfw.make_pair_df((amols))

    return atom_df, amols

df = pd.read_csv('dj1/dj1_data/dj1_clean.csv')

atom_df, amols = write_mm_structures(df)

name = 'dj1/classification/DTree-CHAMPS/Total_predictions/MM/MM_withNHSKRR_withNHsRF_UFF'

dj1_full_class = pd.read_csv('dj1/classification/data_for_classification/DJ1_FULL_class_df.csv')
dj1_four_class = pd.read_csv('dj1/classification/data_for_classification/DJ1_FULL_4class_class_df.csv')

ds = DSCRIBEmodel(id=name)

ds.load_model(model_filename='dj1/classification/DTree-CHAMPS/MM/LMBTR_withNHs_HCS_model.pkl')
ds.targetflag = 'HCS'

ds.test_atom_df = atom_df
ds.test_pair_df = pd.DataFrame()

ds.add_local_many_body_tensor(opt=False, test=True, amols=amols)

target_flags = ['HCS', 'CCS', 'NCS']

print('starting DFT predictions:', datetime.datetime.now())

for i in target_flags:
    ds.load_model(model_filename='dj1/classification/DTree-CHAMPS/MM/LMBTR_withNHs_' + i + '_model.pkl')
    ds.targetflag = i

    pred_df = ds.assign_preds_from_load(ds.model, ds.test_atom_df, write=False, truth=False)

    ds.test_atom_df = pred_df

pred_df.to_csv(ds.id + '_KRR_PREDICTIONS.csv')
print('Finished DFT predictions:', datetime.datetime.now())

### Take output of DSCRIBEmodel as input for ClassifierModel ####
cm = ClassifierModel(id=name)

cm.fulldata_atom_df = pred_df
cm.fulldata_class_df = dj1_full_class[dj1_full_class['Regid'].isin(pred_df['molecule_name'].unique())]

cm.add_dft_to_class_df(cm.fulldata_atom_df, cm.fulldata_class_df, type='predicted_shift')
cm.testids = pred_df['molecule_name'].unique()

print('starting class predictions:', datetime.datetime.now())
cm.load_model(model_filename='dj1/classification/DTree-CHAMPS/withNHs/withNHs_RandomForest_DFT_descriptors_classifier_summary.pkl')

cm.assign_predictions(cm.fulldata_class_df, type='all')

print('finished class predictions:', datetime.datetime.now())

print('script finished:', datetime.datetime.now())
