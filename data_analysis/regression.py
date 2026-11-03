import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import pairplot as pp

def write_yield_data():

    data = pd.read_csv('27D_PC_lodaings.csv.csv')

    condA = {'S1863': 60, 'S1080': 62, 'S1869': 51, 'S5703': 57, 'S1875': 69, 'S5854': 62, 'S5857': 6,
             'S7959': 0, 'S1072': 15, 'S1873': 93, 'S54': 0, 'S8063': 0, 'S5853': 42, 'S5706': 75, 'S5704': 10,
             'S50': 0}

    condB = {'S1863': 48, 'S1080': 45, 'S1869': 57, 'S5703': 57, 'S1875': 21, 'S5854': 0, 'S6018': 0, 'S5857': 50,
             'S7959': 0, 'S1072': 70, 'S1873': 48, 'S54': 0, 'S8063': 0, 'S5853': 1, 'S5706': 15, 'S5704': 0, 'S50': 0}

    condC = {'S1863': 0, 'S1080': 42, 'S1869': 68, 'S5703': 48, 'S1875': 63, 'S5854': 90, 'S6018': 43, 'S5857': 87,
             'S7959': 0, 'S1072': 3, 'S1873': 69, 'S54': 0, 'S8063': 0, 'S5853': 0, 'S5706': 94, 'S5704': 18, 'S50': 0}

    for i in ['A', 'B', 'C']:
        ids = []
        x = []
        acid = []
        nucleo = []
        doni = []
        y = []
        for id in condA.keys():
            ids.append(id)

            ac = acid.append(drow.Rel_acidity.values[0])
            nuc = nucleo.append(drow.Rel_nucleophilicity.values[0])
            don = doni.append(drow.Rel_donicity.values[0])

            x.append([ac, nuc, don])
            eval('y.append(cond' + i + '.get("' + id + '"))')

        df = pd.DataFrame({'Structure': ids, 'Rel_acidity': acid, 'Rel_nucleophilicity': nucleo, 'Rel_donicity': doni,
                           'Yield': y})

        with open('yield_data_cond' + i + '_FOR_SEABORN_PLOT.csv', 'w') as f:
            print(df.to_csv(), file=f)

def mlr_for_yield(yield_data):

    x = np.array(yield_data['X'].values.tolist())
    y = np.array(yield_data['Y'].values)

    model = LinearRegression().fit(x, y)

    r_sq = model.score(x, y)

    print(r_sq)

def mpr_for_yield(yield_data):

    x = np.array(yield_data['X'].values.tolist())
    y = np.array(yield_data['Y'].values)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    degree = {1: 'Linear', 2: 'Squared', 3: 'Cubic'}

    for i in degree.keys():

        poly_model = PolynomialFeatures(degree=i)
        poly_train = poly_model.fit_transform(x_train)
        poly_test = poly_model.fit_transform(x_test)

        lin_model = LinearRegression()
        lin_model.fit(poly_train, y_train)
        pred = lin_model.predict(poly_test)

        r_sq = lin_model.score(poly_train, y_train)

        print(degree.get(i))
        print(f'R2: {r_sq}, MAE: {mean_absolute_error(y_test, pred)}')

#write_yield_data()
A = pd.read_csv('yield_data_condA_FOR_SEABORN_PLOT.csv')
A_y = pd.read_csv('yield_data_condA_NO_NEGATIVES.csv')
B = pd.read_csv('yield_data_condB_NO_NEGATIVES.csv')
C = pd.read_csv('yield_data_condC_NO_NEGATIVES.csv')

ac = A['Rel_acidity'].values
nuc = A['Rel_nucleophilicity'].values
don = A['Rel_donicity'].values

a_yield = A_y['Y']
b_yield = B['Y']
c_yield = C['Y']

#mpr_for_yield(A)
#mpr_for_yield(B)
#mpr_for_yield(C)
