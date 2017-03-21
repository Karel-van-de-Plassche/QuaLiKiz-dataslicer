from IPython import embed
import pandas as pd
import numpy as np
from run_model import QuaLiKizDuoNN, QuaLiKizMultiNN, QuaLiKizNDNN

simple_nns = ['efe_GB',
              'efi_GB',
              'dfe_GB',
              'dfi_GB',
              'ome_GB_less2max',
              'vte_GB_plus_vce_GB',
              'vti_GB_plus_vci_GB']

combo_nns = {'efe_GB2': ['efe_GB_min_efeETG_GB', 'efeETG_GB', lambda x, y: x + y],
             'vte_GB_plus_vce_GB2': ['vte_GB', 'vce_GB', lambda x, y: x + y],
             'vti_GB_plus_vci_GB2': ['vti_GB', 'vci_GB', lambda x, y: x + y]
}

nns = []
for name in simple_nns:
    nn = QuaLiKizNDNN.from_json('nns/nn_' + name + '.json')
    nns.append(nn)

for name, recipe in combo_nns.items():
    nn1 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[0] + '.json')
    nn2 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[1] + '.json')
    nn = QuaLiKizDuoNN(name, nn1, nn2, recipe[2])
    nns.append(nn)
nn = QuaLiKizMultiNN(nns)
if __name__ == '__main__':
    scann = 24
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeffx']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['qx'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    fluxes = nn.get_outputs(**input)
    print(fluxes)
    embed()
