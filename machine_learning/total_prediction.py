import datetime
from model.classifier_model import ClassifierModel
from model.dscribe_model import DSCRIBEmodel
import pandas as pd

dj1_test_atom = pd.read_csv('dj1/classification/data_for_classification/DJ1_exp_test_atom_df.csv')
dj1_test_pair = pd.read_csv('dj1/classification/data_for_classification/DJ1_exp_test_pair_df.csv')
dj1_full_class = pd.read_csv('dj1/classification/data_for_classification/DJ1_FULL_class_df.csv')

four_class = pd.read_csv('dj1/classification/data_for_classification/DJ1_FULL_4class_class_df.csv')

print('initialising script:', datetime.datetime.now())

### Use DSCRIBEmodel to make predictions of acidity, nucleophilicity and donicity values ###
ds = DSCRIBEmodel(id='dj1/classification/DTree-CHAMPS/Total_predictions/MM/withNHsDFT_withNHsClass_DFT')

ds.load_model(model_filename='dj1/classification/KRR_Preds/LMBTR_noNHs_HCS_model.pkl')
ds.targetflag = 'HCS'

ds.test_atom_df = dj1_test_atom
ds.test_pair_df = dj1_test_pair

ds.add_local_many_body_tensor(opt=False, test=True)

target_flags = ['HCS', 'CCS', 'NCS']

print('starting DFT predictions:', datetime.datetime.now())

for i in target_flags:
    ds.load_model(model_filename='dj1/classification/DTree-CHAMPS/MM/LMBTR_withNHs_' + i + '_model.pkl')
    ds.targetflag = i

    pred_df = ds.assign_preds_from_load(ds.model, ds.test_atom_df, write=False)

    ds.test_atom_df = pred_df

pred_df.to_csv(ds.id + '_KRR_PREDICTIONS.csv')
print('Finished DFT predictions:', datetime.datetime.now())

### Take output of DSCRIBEmodel as input for ClassifierModel ####
cm = ClassifierModel(id='dj1/classification/DTree-CHAMPS/Total_predictions/MM/withNHs_DFT')

cm.fulldata_atom_df = pred_df
cm.fulldata_class_df = dj1_full_class[dj1_full_class['Regid'].isin(pred_df['molecule_name'].unique())]

cm.add_dft_to_class_df(cm.fulldata_atom_df, cm.fulldata_class_df, type='predicted_shift')
cm.testids = pred_df['molecule_name'].unique()

print('starting class predictions:', datetime.datetime.now())
cm.load_model(model_filename='dj1/classification/DTree-CHAMPS/withNHs/withNHs_RandomForest_DFT_descriptors_classifier_summary.pkl')

cm.assign_predictions(cm.fulldata_class_df, type='all')

print('finished class predictions:', datetime.datetime.now())

print('script finished:', datetime.datetime.now())
