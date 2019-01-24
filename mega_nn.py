import os
import sys
import copy

import pandas as pd
import numpy as np
from IPython import embed

sys.path.append('../QuaLiKiz-pythontools')
from qlknn.NNDB.model import Network, select_from_candidate_query, get_pure_from_cost_l2_scale, get_from_cost_l2_scale_array, get_pure_from_hyperpar
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN
from qlknn.models.rotdiv import RotDivNN
from qlknn.models.clipping import LeadingFluxNN
from qlknn.models.victor_rule import VictorNN
from qlknn.misc.analyse_names import is_pure, is_flux, is_transport, split_parts, extract_part_names

nn_dict_fakey = {
    'feature_names': ['Machtor'],
    'target_names': ['efeETG_GB_div_efeETG_GB_rot0'],
    'feature_min': {'Machtor': -np.inf},
    'feature_max': {'Machtor': +np.inf},
    'target_min': {'efeETG_GB_div_efeETG_GB_rot0': -np.inf},
    'target_max': {'efeETG_GB_div_efeETG_GB_rot0': +np.inf},
    'prescale_bias': {'Machtor': 0,
                      'efeETG_GB_div_efeETG_GB_rot0': 1},
    'prescale_factor': {'Machtor': 0,
                        'efeETG_GB_div_efeETG_GB_rot0': -1},
    'hidden_activation': [],
    'output_activation': 'none'
}

def combo_func(*args):
    return np.hstack(args)

nn_source = 'NNDB'
nn_source = 'QLKNN-networks'
nn_source = 'RAPTOR_gen3_nets'
rotdiv = True
if nn_source == 'NNDB':
    nN_mn_out = 7

    ITG_list = [get_pure_from_hyperpar(
        'efiITG_GB', nN_mn_out,
        cost_l2_scale=5e-5,
        cost_stable_positive_function='block',
        cost_stable_positive_scale=1e-3,
        cost_stable_positive_offset=-5)
                ]
    ITG_list.append(get_pure_from_hyperpar('efeITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('pfeITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('dfeITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('dfiITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('vteITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('vtiITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('vceITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    ITG_list.append(get_pure_from_hyperpar('vciITG_GB_div_efiITG_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list = [get_pure_from_hyperpar(
        'efeTEM_GB', nN_mn_out,
        cost_l2_scale=5e-5,
        cost_stable_positive_function='block',
        cost_stable_positive_scale=1e-3,
        cost_stable_positive_offset=-5)
                ]
    TEM_list.append(get_pure_from_hyperpar('efiTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('pfeTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('dfeTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('dfiTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('vteTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('vtiTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('vceTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    TEM_list.append(get_pure_from_hyperpar('vciTEM_GB_div_efeTEM_GB', nN_mn_out, cost_l2_scale=5e-5))
    ETG_list = [get_pure_from_hyperpar(
        'efeETG_GB', nN_mn_out,
        cost_l2_scale=5e-5,
        cost_stable_positive_function='block',
        cost_stable_positive_scale=1e-3,
        cost_stable_positive_offset=-5)
                ]
    netgam = get_pure_from_hyperpar('gam_leq_GB', nN_mn_out, cost_l2_scale=2e-5)


    Network_list = ITG_list + TEM_list + ETG_list
    Network_list.append(netgam)
    #{gam_leq_GB}

    combo_target_names = []
    networks = {net.target_names[0]: net.to_QuaLiKizNDNN() for net in Network_list}


    #for net in Network_list:
    #    combo_target_names.extend(net.target_names)
    #.append(gam._target_names, ignore_index=True)
elif nn_source in ['RAPTOR_gen3_nets']:
    from collections import OrderedDict
    networks = OrderedDict()
    for path in os.listdir(nn_source):
        nn = QuaLiKizNDNN.from_json(os.path.join(nn_source, path + '/nn.json'))
        if len(nn._target_names) > 1:
            raise
        else:
            networks[nn._target_names[0]] = nn
    networks['efeETG_GB_div_efeETG_GB_rot0'] = QuaLiKizNDNN(nn_dict_fakey)
elif nn_source in ['QLKNN-networks']:
    from collections import OrderedDict
    networks = OrderedDict()
    for path in os.listdir(nn_source):
        if path.endswith('.json'):
            nn = QuaLiKizNDNN.from_json(os.path.join(nn_source, path))
            if len(nn._target_names) > 1:
                raise
            else:
                networks[nn._target_names[0]] = nn
    if rotdiv:
        networks['efeETG_GB_div_efeETG_GB_rot0'] = QuaLiKizNDNN(nn_dict_fakey)
        # Use the efi rotdiv for efe
        networks['efeITG_GB_div_efeITG_GB_rot0'] = copy.deepcopy(networks['efiITG_GB_div_efiITG_GB_rot0'])
        networks['efeITG_GB_div_efeITG_GB_rot0']._target_names = pd.Series(['efeITG_GB_div_efeITG_GB_rot0'])
        networks['efeTEM_GB_div_efeTEM_GB_rot0'] = copy.deepcopy(networks['efiTEM_GB_div_efiTEM_GB_rot0'])
        networks['efeTEM_GB_div_efeTEM_GB_rot0']._target_names = pd.Series(['efeTEM_GB_div_efeTEM_GB_rot0'])
    else:
        networks = {name: net for name, net in networks.items() if 'rot' not in name}
if rotdiv and nn_source is not 'QLKNN-networks':
    nN_mn_out = 8
    rot_ITG_list = [get_pure_from_hyperpar(
            'efiITG_GB_div_efiITG_GB_rot0', nN_mn_out,
            cost_l2_scale=5e-4,
            cost_stable_positive_function='block',
            cost_stable_positive_scale=1e-2,
            cost_stable_positive_offset=-5)
                    ]
    rot_ITG_list.append(get_pure_from_hyperpar('efeITG_GB_div_efeITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('pfeITG_GB_div_pfeITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('dfeITG_GB_div_dfeITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('dfiITG_GB_div_dfiITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('vteITG_GB_div_vteITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('vtiITG_GB_div_vtiITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('vceITG_GB_div_vceITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_ITG_list.append(get_pure_from_hyperpar('vciITG_GB_div_vciITG_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list = [get_pure_from_hyperpar(
            'efeTEM_GB_div_efeTEM_GB_rot0', nN_mn_out,
            cost_l2_scale=5e-4,
            cost_stable_positive_function='block',
            cost_stable_positive_scale=1e-2,
            cost_stable_positive_offset=-5)
                    ]
    rot_TEM_list.append(get_pure_from_hyperpar('efiTEM_GB_div_efiTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('pfeTEM_GB_div_pfeTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('dfeTEM_GB_div_dfeTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('dfiTEM_GB_div_dfiTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('vteTEM_GB_div_vteTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('vtiTEM_GB_div_vtiTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('vceTEM_GB_div_vceTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_TEM_list.append(get_pure_from_hyperpar('vciTEM_GB_div_vciTEM_GB_rot0', nN_mn_out, cost_l2_scale=5e-4))
    rot_Network_list = rot_ITG_list + rot_TEM_list
    rot_networks = {net.target_names[0]: net.to_QuaLiKizNDNN() for net in rot_Network_list}
    networks.update(rot_networks)
# Match all div networks with their leading fluxes
from functools import reduce
for target_name in list(networks.keys()):
    if is_transport(target_name) and not is_pure(target_name):
        target, op, leading = split_parts(target_name)
        if op != '_div_':
            raise
        nn_norot = QuaLiKizComboNN(pd.Series(target),
                              [networks.pop(target_name), networks[leading]],
                              lambda x, y: x * y)
        if rotdiv and 'ETG' not in target:
            rotname = target + '_div_' + target + '_rot0'
            networks[target] = RotDivNN(nn_norot, networks.pop(rotname))
        else:
            networks[target] = nn_norot
if rotdiv:
    for target in ['efiITG_GB', 'efeTEM_GB', 'efeETG_GB']:
        rotname = target + '_div_' + target + '_rot0'
        networks[target] = RotDivNN(networks.pop(target), networks.pop(rotname), allow_negative=False)

gam = networks.pop('gam_leq_GB')
nets = list(networks.values())
combo_target_names = list(networks.keys())

combo_nn = QuaLiKizComboNN(pd.Series(combo_target_names), nets, combo_func)

#vic_nn = VictorNN(combo_nn, gam)
nn = LeadingFluxNN.add_leading_flux_clipping(combo_nn)
qlknn_9D_feature_names = [
        "Zeff",
        "Ati",
        "Ate",
        "An",
        "q",
        "smag",
        "x",
        "Ti_Te",
        "logNustar",
    ]

raptor_order = [
    'efeETG_GB'              ,# 1
    'efeITG_GB',# 2
    'efeTEM_GB',# 3
    'efiITG_GB',# 4
    'efiTEM_GB',# 5
    'pfeITG_GB',# 6
    'pfeTEM_GB',# 7
    'dfeITG_GB',# 8
    'dfeTEM_GB',# 9
    'vteITG_GB',# 10
    'vteTEM_GB',# 11
    'vceITG_GB',# 12
    'vceTEM_GB',# 13
    'dfiITG_GB',# 14
    'dfiTEM_GB',# 15
    'vtiITG_GB',# 16
    'vtiTEM_GB',# 17
    'vciITG_GB',# 18
    'vciTEM_GB',# 19
    'gam_leq_GB',
    ];

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
    input['Machtor']  = np.full_like(input['Ati'], 0.4)
    #low_bound = np.array([[0 if ('ef' in name) and (not 'div' in name) else -np.inf for name in nn._target_names]]).T
    #low_bound = pd.DataFrame(index=nn._target_names, data=low_bound)
    low_bound = None
    high_bound = None

    #print('Seperate')
    #print(ITG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(TEM.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(ETG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    print('Combo NN')
    combo_flux = combo_nn.get_output(input, safe=True, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    leading_flux = nn.get_output(input, safe=True, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    print(leading_flux.loc[:, raptor_order[:-1]])
    ['efeETG_GB', 'efiITG_GB', 'efeTEM_GB', 'vtiTEM_GB', 'dfiTEM_GB',
       'vciITG_GB', 'dfeTEM_GB', 'vceTEM_GB', 'vciTEM_GB', 'pfeTEM_GB',
       'vteITG_GB', 'vteTEM_GB', 'vceITG_GB', 'efeITG_GB', 'pfeITG_GB',
       'dfeITG_GB', 'efiTEM_GB', 'vtiITG_GB', 'dfiITG_GB']
    combined_fluxes = OrderedDict([
        ('efe_GB', leading_flux[['efeETG_GB', 'efeITG_GB', 'efeTEM_GB']].sum(axis=1)),
        ('efeETG_GB', leading_flux[['efeETG_GB']].sum(axis=1)),
        ('efi_GB', leading_flux[['efiITG_GB', 'efiTEM_GB']].sum(axis=1)),
        ('pfe_GB', leading_flux[['pfeITG_GB', 'pfeTEM_GB']].sum(axis=1)),
        ('dfe_GB', leading_flux[['dfeITG_GB', 'dfeTEM_GB']].sum(axis=1)),
        ('vte_GB', leading_flux[['vteITG_GB', 'vteTEM_GB']].sum(axis=1)),
        ('vce_GB', leading_flux[['vceITG_GB', 'vceTEM_GB']].sum(axis=1)),
        ('dfi_GB', leading_flux[['dfiITG_GB', 'dfiTEM_GB']].sum(axis=1)),
        ('vti_GB', leading_flux[['vtiITG_GB', 'vtiTEM_GB']].sum(axis=1)),
        ('vci_GB', leading_flux[['vciITG_GB', 'vciTEM_GB']].sum(axis=1)),
    ])
    combof = pd.DataFrame(combined_fluxes)
    print(combof)
    embed()
    #input['gammaE'] = np.full_like(input['Ati'], 0.1)
    #print('Victor NN')
    #print(vic_nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print('Clipped NN')
    #fluxes = nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    #print(fluxes)
