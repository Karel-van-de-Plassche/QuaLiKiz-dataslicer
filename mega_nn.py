import os
import sys

import pandas as pd
import numpy as np
from IPython import embed

from qlknn.NNDB.model import Network, select_from_candidate_query, get_pure_from_cost_l2_scale, get_from_cost_l2_scale_array
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN
from qlknn.models.clipping import LeadingFluxNN
from qlknn.models.victor_rule import VictorNN


def combo_func(*args):
    return np.hstack(args)

nn_source = 'NNDB'
nn_source = 'RAPTOR_gen3_nets'
if nn_source == 'NNDB':
    ITG_list = [get_pure_from_cost_l2_scale('efiITG_GB', 5e-5)]
    ITG_list.append(get_from_cost_l2_scale_array('efeITG_GB', '{5e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('pfeITG_GB', '{5e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('dfeITG_GB', '{5e-5, 8e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('dfiITG_GB', '{5e-5, 8e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('vteITG_GB', '{5e-5, 8e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('vtiITG_GB', '{5e-5, 8e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('vceITG_GB', '{5e-5, 8e-5}'))
    ITG_list.append(get_from_cost_l2_scale_array('vciITG_GB', '{5e-5, 8e-5}'))
    TEM_list = [get_pure_from_cost_l2_scale('efeTEM_GB', 5e-5)]
    TEM_list.append(get_from_cost_l2_scale_array('efiTEM_GB', '{5e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('pfeTEM_GB', '{5e-5, 1e-4}'))
    TEM_list.append(get_from_cost_l2_scale_array('dfeTEM_GB', '{5e-5, 8e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('dfiTEM_GB', '{5e-5, 8e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('vteTEM_GB', '{5e-5, 8e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('vtiTEM_GB', '{5e-5, 8e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('vceTEM_GB', '{5e-5, 8e-5}'))
    TEM_list.append(get_from_cost_l2_scale_array('vciTEM_GB', '{5e-5, 8e-5}'))
    ETG_list = [get_pure_from_cost_l2_scale('efeETG_GB', 5e-5)]
    Network_list = ITG_list + TEM_list + ETG_list
    #{gam_leq_GB}
    netgam = get_pure_from_cost_l2_scale('gam_leq_GB', 1e-5)

    gam = netgam.to_QuaLiKizNN() #1e-5
    nets = [net.to_QuaLiKizNN() for net in Network_list]
    combo_target_names = []
    for net in Network_list:
        combo_target_names.extend(net.target_names)
    #.append(gam._target_names, ignore_index=True)
elif nn_source == 'RAPTOR_gen3_nets':
    from collections import OrderedDict
    networks = OrderedDict()
    for path in os.listdir(nn_source):
        nn = QuaLiKizNDNN.from_json(os.path.join(nn_source, path + '/nn.json'))
        networks[tuple(nn._target_names)] = nn
    gam = networks.pop(('gam_leq_GB',))
    nets = list(networks.values())
    combo_target_names = [key[0] for key in networks.keys()]

combo_nn = QuaLiKizComboNN(pd.Series(combo_target_names), nets, combo_func)
vic_nn = VictorNN(combo_nn, gam)
nn = LeadingFluxNN.add_leading_flux_clipping(vic_nn)

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
    input['logNustar']  = np.full_like(input['Ati'], np.log10(0.009995))
    input['x']  = np.full_like(input['Ati'], 0.449951)
    #low_bound = np.array([[0 if ('ef' in name) and (not 'div' in name) else -np.inf for name in nn._target_names]]).T
    #low_bound = pd.DataFrame(index=nn._target_names, data=low_bound)
    low_bound = None
    high_bound = None

    #print('Seperate')
    #print(ITG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(TEM.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(ETG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    print('Combo NN')
    print(combo_nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    input['gammaE'] = np.full_like(input['Ati'], 0.1)
    print('Victor NN')
    print(vic_nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    print('Clipped NN')
    fluxes = nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    print(fluxes)
    embed()
