import os
import sys

import pandas as pd
import numpy as np
from IPython import embed

from qlknn.NNDB.model import Network
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN

ITG = Network.get_by_id(834).to_QuaLiKizNN()
TEM = Network.get_by_id(848).to_QuaLiKizNN()
ETG = Network.get_by_id(740).to_QuaLiKizNN()
def combo_func(*args):
    return np.hstack(args)
combo_target_names = ITG._target_names.append(TEM._target_names, ignore_index=True).append(ETG._target_names, ignore_index=True)
nn = QuaLiKizComboNN(combo_target_names, [ITG, TEM, ETG], combo_func)

if __name__ == '__main__':
    scann = 24
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeff']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['q'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['logNustar']  = np.full_like(input['Ati'], np.log10(0.009995))
    input['x']  = np.full_like(input['Ati'], 0.449951)
    low_bound = np.array([[0 if ('ef' in name) and (not 'div' in name) else -np.inf for name in nn._target_names]]).T
    low_bound = pd.DataFrame(index=nn._target_names, data=low_bound)
    high_bound = None

    fluxes = nn.get_output(input, clip_high=False, clip_low=True, high_bound=high_bound, low_bound=low_bound)
    print(fluxes)
    print(ITG.get_output(input))
    print(TEM.get_output(input))
    print(ETG.get_output(input))
    embed()
