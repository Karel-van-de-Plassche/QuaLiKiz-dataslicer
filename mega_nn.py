from IPython import embed
import pandas as pd
import numpy as np
import os
import sys
from qlknn.NNDB.model import Network
from qlknn.models.ffnn import QuaLiKizComboNN, QuaLiKizNDNN

ITG_nn = Network.get_by_id(423).to_QuaLiKizNN()
TEM_nn = Network.get_by_id(166).to_QuaLiKizNN()
ETG_nn = Network.get_by_id(331).to_QuaLiKizNN()

nns = [
    ITG_nn,
    TEM_nn,
    ETG_nn
]

target_names = []
for nn in nns:
    target_names.extend(nn._target_names)

nn = QuaLiKizComboNN(target_names,
                     nns,
                     lambda *x: np.hstack(x))
#
#for name, recipe in combo_nns.items():
#    nn1 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[0] + '.json')
#    nn2 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[1] + '.json')
#    nn = QuaLiKizDuoNN(name, nn1, nn2, recipe[2])
#    nns.append(nn)

#nn = QuaLiKizMultiNN(nns)
if __name__ == '__main__':
    scann = 24
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    #input['Zeffx']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['qx'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    fluxes = nn.get_output(input, safe=True)
    print(fluxes)
    embed()
