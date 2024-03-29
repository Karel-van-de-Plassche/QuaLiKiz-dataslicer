import os
import sys
from warnings import warn

import pandas as pd
import numpy as np
from IPython import embed
from collections import OrderedDict

#sys.path.append('../QuaLiKiz-pythontools')
sys.path.append('../QLKNN-develop/qlknn/models/')
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN
from qlknn.models.rotdiv import RotDivNN
from qlknn.models.clipping import LeadingFluxNN
from qlknn.models.victor_rule import VictorNN
from qlknn.misc.analyse_names import is_pure, is_flux, is_transport, split_parts, extract_part_names

def combo_func(*args):
    return np.hstack(args)

class Summer():
    def __init__(self, idx):
        self.idx = idx

    def sum(self, *outp):
        if isinstance(outp, pd.DataFrame):
            raise NotImplementedError
        else:
            if outp[0].ndim == 1:
                return np.atleast_1d(np.sum(outp[0][self.idx]))
            else:
                return np.atleast_2d(np.sum(outp[0][:,self.idx], axis=1)).T

nn_source = 'NNDB'
nn_source = 'QLKNN-networks'
nn_source = 'QLKNN-fortran'
if nn_source == 'NNDB':
    from qlknn.NNDB.model import Network, select_from_candidate_query, get_pure_from_cost_l2_scale, get_from_cost_l2_scale_array, get_pure_from_hyperpar
if nn_source == 'QLKNN-fortran':
    from qlknn.models.qlknn_fortran import QuaLiKizFortranNN
    print('Loading FORTRAN network')
    fnn = QuaLiKizFortranNN('../QLKNN-fortran/lib/src/qlknn-hyper-namelists')
    fnn.opts.merge_modes = 0
    fnn.opts.force_evaluate_all = 1
    combo_fnn = QuaLiKizFortranNN('../QLKNN-fortran/lib/src/qlknn-hyper-namelists')
    combo_fnn.opts.merge_modes = 1
    combo_nn = QuaLiKizComboNN(combo_fnn._target_names + fnn._target_names, [combo_fnn, fnn], combo_func)
    # Nested-python ComboNetwork way
    #combo_nns = []
    #target_names = []
    #summers = []
    #idxs = {}
    #prefixes = ['efe', 'efi', 'pfe', 'dfe', 'vte', 'vce', 'dfi', 'vti', 'vci']
    #for prefix in prefixes:
    #    fnn.opts.force_evaluate_all = True
    #    this_idx = [fnn._target_names.index(name) for name in fnn._target_names if name.startswith(prefix)]
    #    summers.append(Summer(this_idx))
    #    print('Creating', prefix, 'network')
    #    combo_nns.append(QuaLiKizComboNN(pd.Series([prefix + '_GB']), [fnn], summers[-1].sum))
    #    target_names.append(prefix + '_GB')
    #combo_nn = QuaLiKizComboNN(target_names + fnn._target_names, combo_nns + [fnn], combo_func)
    nn = combo_nn
elif nn_source == 'NNDB':
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

elif nn_source in ['QLKNN-networks']:
    from collections import OrderedDict
    networks = OrderedDict()
    nn_folder = nn_source
    if len(os.listdir(nn_folder)) == 0:
        warn("QLKNN-network folder '{!s}' empty".format(nn_folder))
    for path in os.listdir(nn_folder):
        if path.endswith('.json'):
            nn = QuaLiKizNDNN.from_json(os.path.join(nn_folder, path))
            if len(nn._target_names) > 1:
                raise Exception('Multi-target NN! Not sure what to do..')
            else:
                networks[nn._target_names[0]] = nn
    networks = {name: net for name, net in networks.items()}
    if len(networks) == 0:
        warn("Did not find any networks in '{!s}'! Does it contain the JSONs?")

if nn_source != 'QLKNN-fortran':
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
            networks[target] = nn_norot

    if 'gam_leq_GB' in networks:
        gam_name = 'gam_leq_GB'
    else:
        raise Exception('No gam network found!')
    gam = networks.pop(gam_name)
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
    input['Te']  = np.full_like(input['Ati'], 1.)
    input['Zeff']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['q'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['logNustar']  = np.full_like(input['Ati'], np.log10(0.009995))
    input['x']  = np.full_like(input['Ati'], 0.449951)
    input['gammaE_QLK']  = np.full_like(input['Ati'], 0.4)
    #low_bound = np.array([[0 if ('ef' in name) and (not 'div' in name) else -np.inf for name in nn._target_names]]).T
    #low_bound = pd.DataFrame(index=nn._target_names, data=low_bound)
    low_bound = None
    high_bound = None

    #print('Seperate')
    #print(ITG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(TEM.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print(ETG.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    print('Combo NN')
    leading_flux = nn.get_output(input, safe=True, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    if combo_nn:
        combo_flux = combo_nn.get_output(input, safe=True, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
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
        print(combo_flux)
    print(leading_flux.loc[:, raptor_order[:-1]])
    embed()
    #input['gammaE'] = np.full_like(input['Ati'], 0.1)
    #print('Victor NN')
    #print(vic_nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound))
    #print('Clipped NN')
    #fluxes = nn.get_output(input, clip_high=False, clip_low=False, high_bound=high_bound, low_bound=low_bound)
    #print(fluxes)
