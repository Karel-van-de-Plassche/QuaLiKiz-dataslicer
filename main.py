"""
Copyright Dutch Institute for Fundamental Energy Research (2016)
Contributors: Karel van de Plassche (karelvandeplassche@gmail.com)
License: CeCILL v2.1
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
import time
from collections import OrderedDict
from math import ceil
from itertools import repeat, product
import pprint
#
#sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '.'))))
#print(sys.path)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from IPython import embed
from bokeh.plotting import figure, show, reset_output, Figure
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox
from bokeh.models.widgets import Button, Div
from bokeh.models import HoverTool, Band
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, CustomJS, Legend, Line, Circle
from bokeh.palettes import Set1 as sepcolor
from bokeh.palettes import Plasma256
#TODO: Add sane checking if loading failed and why
import sys
sys.path.append('../QLKNN-develop')
sys.path.append('../QuaLiKiz-pythontools')
sys.path.append('./bokeh-ion-rangeslider')
sys.path.append('../QLKNN-develop/qlknn/models/')

from bokeh_ion_rangeslider import IonRangeSlider
try:
    ModuleNotFoundError
except:
    ModuleNotFoundError = ImportError
plot_nn = True
try:
    from qlknn.models.ffnn import QuaLiKizNDNN
    from qlknn.models.victor_rule import gammaE_QLK_to_gammaE_GB, gammaE_GB_to_gammaE_QLK, VictorNN
except ModuleNotFoundError:
    print("Could not import QuaLiKizNDNN")
    plot_nn = False
try:
    from qlknn.models.qlknn_fortran import QuaLiKizFortranNN
except ModuleNotFoundError:
    print("Could not import QuaLiKizFortranNN")
if plot_nn:
    try:
        import mega_nn
    except:
        print('Could not import mega_nn')
        plot_nn = False
from qualikiz_tools.misc.conversion import calc_nustar_from_parts, calc_te_from_nustar, calc_puretor_gradient, calc_epsilon_from_parts, calc_puretor_gradient

def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]

def extract_plotdata(sel_dict):
    start = time.time()
    slice_ = ds.sel(**sel_dict, method='nearest', tolerance=1e-4)
    slice_.load()
    #slice_sep = ds_sep.sel(**sel_dict)
    #slice_grow = ds_grow.sel(**sel_dict)
    #slice_ = xr.merge([slice_, slice_grow])
    if 'nions' in slice_.dims:
        slice_ = slice_.sel(nions=0)
    if xaxis_name == rotvar:
        slice_flux = slice_[fluxlike_vars].reset_coords(drop=True)
        columns = [k for k in slice_flux.variables if k not in slice_flux.dims]
        data = [slice_flux._variables[k].set_dims([]).values.reshape(-1) for k in columns]
        df_flux = pd.DataFrame(OrderedDict(zip(columns, data)), index=[ds.attrs[xaxis_name]])
    elif rotvar not in sel_dict or np.isclose(sel_dict[rotvar], 0) or rotvar in ds[fluxlike_vars[0]].dims:
        df_flux = slice_[fluxlike_vars].reset_coords(drop=True).to_dataframe()
    else:
        df_flux = pd.DataFrame()
    df_flux.index.name = 'xaxis'
    df_flux.reset_index(inplace=True)

    try:
        slice_rot = ds_rot.sel(**{k: v for k, v in sel_dict.items() if k in ds_rot.dims},
                               method='nearest', tolerance=1e-4)
        for k, v in sel_dict.items():
            if k not in ds_rot.dims:
                if not np.isclose(v, ds_rot.attrs[k]):
                    raise KeyError
    except (KeyError, NameError) as ee:
        df_rot = pd.DataFrame()
    else:
        if 'nions' in slice_rot.dims:
            slice_rot = slice_rot.sel(nions=0)
        slice_rot = slice_rot[fluxlike_vars].reset_coords(drop=True)
        try:
            df_rot = slice_rot.to_dataframe()
        except ValueError: #0D slice
            columns = [k for k in slice_rot.variables if k not in slice_rot.dims]
            data = [slice_rot._variables[k].set_dims([]).values.reshape(-1) for k in columns]
            df_rot = pd.DataFrame(OrderedDict(zip(columns, data)), index=[ds_rot.attrs[xaxis_name]])

        no_df_cols = [col for col in df_rot.columns if 'df' not in col]
        df_rot[no_df_cols] = df_rot[no_df_cols] * 3

        df_rot.index.name = 'xaxis'
        df_rot.reset_index(inplace=True)

    if plot_nn:
        df_nns = []
        for nn_name, nn in nns.items():
        #xaxis = df_flux['xaxis']
            xaxis = pd.Series(ds[xaxis_name].values, name=xaxis_name)
            nn_xaxis = np.linspace(xaxis.iloc[0], xaxis.iloc[-1], 200)

            inp = pd.DataFrame({name: float(slice_[name]) for name in nn._feature_names if name != xaxis_name and name in slice_}, index=[0])
            input = pd.DataFrame({xaxis_name: np.linspace(xaxis.iloc[0], xaxis.iloc[-1], 60)}).join(inp).fillna(method='ffill')
            for name in nn._feature_names:
                if name not in input:
                    if name in ds.attrs:
                        input[name] = ds.attrs[name]
                    elif name == 'logNustar' and 'Nustar' in ds.attrs:
                        input[name] = np.log10(ds.attrs['Nustar'])

            try:
                is_fortranNN = isinstance(nn, QuaLiKizFortranNN)
            except NameError:
                is_fortranNN = False
            if plot_victor and ((hasattr(nn, '_internal_network') and isinstance(nn._internal_network, VictorNN)) or # Has victor rule
                                is_fortranNN): #Is Fortran NN
                vars = pd.DataFrame()
                for name in ['Zeff', 'ne', 'Nustar', 'logNustar', 'q', 'Ro', 'Rmin', 'x']:
                    if name in input:
                        vars[name] = input[name]
                    elif name in ds.attrs:
                        vars[name] = ds.attrs[name]
                if 'logNustar' in vars:
                    vars['Nustar'] = 10**vars.pop('logNustar')
                Te = calc_te_from_nustar(*[vars[name] for name in ['Zeff', 'ne', 'Nustar', 'q', 'Ro', 'Rmin', 'x']])
                if 'gammaE_QLK' in sel_dict:
                    gammaE_QLK = sel_dict['gammaE_QLK']
                elif xaxis_name == 'gammaE_QLK':
                    gammaE_QLK = input.pop('gammaE_QLK')

                if hasattr(nn, '_internal_network') and isinstance(nn._internal_network, VictorNN):
                    df_nn = nn.get_output(input)
                elif isinstance(nn, QuaLiKizFortranNN):
                    input['Te'] = Te
                    vars['Ai1'] = ds.attrs['Ai'][0]
                    df_nn = nn.get_output(input, R0=ds.attrs['Ro'], a=ds.attrs['Rmin'], A1=vars['Ai1'])
            else:
                df_nn = nn.get_output(input)
            if gam_leq_nns[nn_name] is not None:
                df_gam_leq = gam_leq_nns[nn_name].get_output(input)
                gam_leq_cols = [col for col in df_gam_leq.columns if col.startswith('gam_leq')]
                df_nn[gam_leq_cols] = df_gam_leq[gam_leq_cols].clip(lower=0)
            df_nn.drop([name for name in nn._target_names if not name in fluxlike_vars], axis=1, inplace=True)
            not_there = [var for var in fluxlike_vars if var not in df_nn.columns]
            if plot_nn_eb:
                cols_with_eb = [col[:-3] for col in df_nn.columns if col.endswith('_EB')]
                for col in cols_with_eb:
                    df_nn[col + '_LEB'] = df_nn.loc[:, col] - df_nn.loc[:, col + '_EB']
                    df_nn[col + '_UEB'] = df_nn.loc[:, col] + df_nn.loc[:, col + '_EB']
                    df_nn.drop(col + '_EB', axis=1, inplace=True)
                not_there += [var + '_LEB' for var in fluxlike_vars if var not in df_nn.columns]
                not_there += [var + '_UEB' for var in fluxlike_vars if var not in df_nn.columns]
                df_nn['efiITG_GB_EB'] = 1
            for var in set(fluxlike_vars) - set(df_nn.columns):
                df_nn[var] = np.NaN
            if plot_nn_eb:
                for var in ((set([var + '_LEB' for var in fluxlike_vars]) |
                             set([var + '_UEB' for var in fluxlike_vars])) -
                            set(df_nn.columns)
                            ):
                    df_nn[var] = np.NaN

            df_nn.columns = [nn_name + '_' + name for name in df_nn.columns]
            #df_nn.index = (input[xaxis_name])
            #df_nn.index.name = 'xaxis'
            #df_nn.reset_index(inplace=True)
            df_nns.append(df_nn)
        df_nns = pd.concat(df_nns, axis=1)
        df_nns['xaxis'] = input[xaxis_name]
    else:
        df_nns = pd.DataFrame()

    if plot_freq:
        df_freq = slice_[freq_vars].reset_coords(drop=True).to_dataframe()
        df_freq.reset_index(inplace=True)
    else:
        df_freq = pd.DataFrame()
    #df_freq.index.set_names('xaxis', level='kthetarhos', inplace=True)

    if fake_rotvar and slice_['gammaE_QLK'] != 0:
        df_flux = pd.DataFrame()

    return df_flux, df_freq, df_nns, df_rot

def swap_x(attr, old, new):
    global xaxis_name, flux_source, freq_source, nn_source
    try:
        xaxis_name = new[0]
        old_xaxis_name = old[0]
    except TypeError:
        return
    for source in [flux_source, freq_source, nn_source]:
        source.data = {name: [] for name in source.column_names}
    for fig in fluxfigs.values():
        if xaxis_name == 'gammaE':
            fig.x_range.start = float(max(ds[xaxis_name]))
            fig.x_range.end = float(min(ds[xaxis_name]))
        else:
            fig.x_range.start = float(min(ds[xaxis_name]))
            fig.x_range.end = float(max(ds[xaxis_name]))
        fig.xaxis.axis_label = xaxis_name
    updater(None, None, None)

def read_sliders():
    sel_dict = {}
    for name, slider in slider_dict.items():
        if name != xaxis_name:
            sel_dict[name] = slider.value[0]
    return sel_dict

def timer(msg, start):
    print (msg, str(time.time() - start))

def updater(attr, old, new):
    start = time.time()
    try:
        # This will (silently) fail when the sliders are not initialized yet
        sel_dict = read_sliders()
    except TypeError:
        return

    df_flux, df_grow, df_nn, df_rot = extract_plotdata(sel_dict)
    for source, df in zip([flux_source, freq_source, nn_source, rot_source], [df_flux, df_freq, df_nn, df_rot]):
        if len(df) == 0:
            source.data = {name: [] for name in source.column_names}
        else:
            source.data = df.to_dict('list')

def get_nn_scan_dims(nn, scan_dims):
    nn_scan_dims = []
    for dim in scan_dims:
        if dim in nn._feature_names.values:
            if nn.feature_min[dim] != nn.feature_max[dim]:
                nn_scan_dims.append(dim)
    return nn_scan_dims

def print_slice():
    slider_dict = read_sliders()
    text = pprint.pformat(slider_dict, indent=1, width=80)
    text = "&nbsp;".join(text.split(" "))
    text = "<br />".join(text.split("\n"))
    print_slice_text.text = text

def unique_ordered(seq, idfun=None):
    """ Remove duplicate elements and preserve ordering from sequence """
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result



############################################################
# Load dataset                                             #
############################################################
import socket
if socket.gethostname().startswith('rs'):
    root_dir = '/Rijnh/Shares/Departments/Fusiefysica/IMT/karel'
else:
    root_dir = '../qlk_data'
ds_to_plot = '9D'
#ds_to_plot = 'rot_one'
#ds_to_plot = 'bart'
constrain_to_rot = False
if ds_to_plot == '9D':
    ds = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.combo.nions0.nc.1'))
    ds_rot = xr.open_dataset(os.path.join(root_dir, 'rot_three.nc.1'))
    ds_rot.attrs['logNustar'] = np.log10(ds_rot.attrs['Nustar'])
    ds_grow = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.grow.nc'))
    ds_grow = ds_grow.drop([x for x in ds_grow.coords if x not in ds_grow.dims and x not in ['Zi']])
    ds = ds.merge(ds_grow)
    ds.attrs['ne'] = ds.attrs.pop('Nex')
    if constrain_to_rot:
        dummy_var = 'efe_GB'
        rot_dims = ds_rot[dummy_var].dims
        ds = ds.sel({dim: ds_rot[dim].values for dim in rot_dims if dim != 'Machtor'},
                    tolerance=1e-4, method='nearest')
        ds = ds.sel({'Zeff': ds_rot.attrs['Zeff'], 'Nustar': ds_rot.attrs['Nustar']},
                    tolerance=1e-4, method='nearest')

elif ds_to_plot == 'rot_one':
    ds = xr.open_dataset(os.path.join(root_dir, 'rot_one.nc'))
elif ds_to_plot == 'bart':
    ds = xr.open_dataset(os.path.join(root_dir, '../NN-data/qlk_run/qlk_run.nc'))
    ds['Ati0'] = ds['Ati']
    ds = ds.swap_dims({'Ati': 'Ati0'})

#ds = ds.drop([x for x in ds.coords if x not in ds.dims and x not in ['Zi']])
if 'Nustar' in ds.dims:
    ds['logNustar'] = np.log10(ds['Nustar'])
    ds = ds.swap_dims({'Nustar': 'logNustar'})
elif 'Nustar' in ds.coords:
    ds.coords['logNustar'] = np.log10(ds['Nustar'])
if 'Zeffx' in ds.dims:
    ds = ds.rename({'Zeffx': 'Zeff'})
if 'qx' in ds.dims:
    ds = ds.rename({'qx': 'q'})

#ds_sep = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.sep.nc.1'))
#ds_sep = ds_sep.drop([x for x in ds_sep.coords if x not in ds_sep.dims and x not in ['Zi']])
#ds = xr.open_dataset('4D.nc3')

plot_nn = plot_nn and True
plot_freq = True
plot_df = False
plot_sepflux = True
plot_victor = True
rotvar = 'gammaE_QLK'
plot_full = True
sepflux_names = ['ITG', 'TEM']

style = 'heat'
style = 'particle'
style = 'diffusivity'
style = 'thermodiffusion'
style = 'convection'
style = 'TEM'
style = 'all'
#style = 'custom'
if style == 'heat':
    plot_ef = True
    plot_pf = False
    plot_grow = False
    sepflux_names.insert(0, 'ETG')
    plot_df = False
    plot_vt = False
    plot_vc = False
elif style == 'particle':
    plot_ef = False
    plot_pf = True
    plot_grow = True
    plot_freq = False
    plot_df = False
    plot_vt = False
    plot_vc = False
elif style == 'diffusivity':
    plot_ef = False
    plot_pf = False
    plot_grow = False
    plot_df = True
    plot_vt = False
    plot_vc = False
elif style == 'thermodiffusion':
    plot_ef = False
    plot_pf = False
    plot_grow = False
    plot_df = False
    plot_vt = True
    plot_vc = False
elif style == 'convection':
    plot_ef = False
    plot_pf = False
    plot_grow = False
    plot_df = False
    plot_vt = False
    plot_vc = True
elif style in ['ETG', 'ITG', 'TEM']:
    plot_ef = True
    plot_freq = False
    plot_grow = False
    plot_full = False
    if style != 'ETG':
        plot_pf = True
        plot_df = True
        plot_vt = True
        plot_vc = True
    else:
        plot_pf = False
        plot_df = False
        plot_vt = False
        plot_vc = False
    sepflux_names = [style]
elif style == 'all':
    plot_ef = True
    plot_freq = False
    plot_grow = False
    plot_full = False
    plot_pf = True
    plot_df = True
    plot_vt = True
    plot_vc = True
    sepflux_names.insert(0, 'ETG')
elif style == 'custom':
    plot_ef = True
    plot_pf = False
    plot_freq = False
    plot_grow = False
    plot_full = True
    sepflux_names.insert(0, 'ETG')
    plot_df = False
    plot_vt = False
    plot_vc = False
else:
    raise Exception('Style {!s} not defined'.format(style))
norm = '_GB'
plot_nn_eb = True
show_legend = True

if plot_full:
    flux_suffixes = ['']
else:
    flux_suffixes = []

if plot_sepflux:
    flux_suffixes += [name for name in sepflux_names]

if plot_nn:
    #nn = QuaLiKizNDNN.from_json('nn.json')
    #from qlknn.models.committee import QuaLikizCommitteeNN
    #const_dict = {'Machtor': 0, 'alpha': 0, 'Autor':0}
    #nns = [QuaLiKizLessDNN.from_json('./nn{0:03d}.json'.format(ii), const_dict=const_dict, Zi=[1, 6]) for ii in range(1, 11)]
    #nn1 = QuaLikizCommitteeNN(nns)
    nn0 = mega_nn.nn
    #from qlknn.models.kerasmodel import Philipp7DNN
    #nn1 = Philipp7DNN.from_files('CGNN_L2_1000_2.h5', 'training_gen3_7D_nions0_flat_filter8.csv', GB_scale_length=3.)
    nns = OrderedDict([('gen4', nn0)])
    #nns = OrderedDict([('gen_4', nn0)])
    #nn1 = QuaLiKizFortranNN('/home/karel/QLKNN-fortran/lib')
    #nns = OrderedDict([('gen_3', nn0), ('JETTO', nn1)])
else:
    nns = {}

nondims = ['nions', 'numsols', 'kthetarhos', 'ecoefs', 'numicoefs', 'ntheta']
scan_dims = [name for name in ds.dims if name not in nondims]
flux_vars = []
for pre in ['ef', 'pf', 'df', 'vt', 'vc']:
    if (
        (pre == 'ef' and not plot_ef) or
        (pre == 'pf' and not plot_pf) or
        (pre == 'df' and not plot_df) or
        (pre == 'vt' and not plot_vt) or
        (pre == 'vc' and not plot_vc)
       ):
        continue
    for suff in flux_suffixes:
        for species in ['i', 'e']:
            if ((suff == 'ETG' and not (pre == 'ef' and species == 'e'))):
                continue
            flux_vars.append((pre, species, suff, norm))
if plot_freq:
    freq_vars = [name for name in ds.data_vars if any(var in name for var in ['ome' + norm, 'gam' + norm])]
else:
    freq_vars = []

if plot_grow:
    flux_vars.append(('gam', '_leq', '', norm))
fluxlike_vars = [''.join(var) for var in flux_vars]
ds = ds.drop([name for name in ds.data_vars if name not in freq_vars + fluxlike_vars])

try:
    is_fortranNN = isinstance(nn, QuaLiKizFortranNN)
except NameError:
    is_fortranNN = False

fake_rotvar = True
if plot_victor:
    for nn_name, nn in nns.items():
        if is_fortranNN:
            nn.opts.apply_victor_rule = True
    if rotvar in ds.coords:
        fake_rotvar = False
        try:
            if rotvar in ds_rot.coords:
                raise Exception('Two datasets with {!s}'.format(rotvar))
        except NameError:
            pass
    else:
        try:
            if rotvar in ds_rot.coords:
                fake_rotvar = False
                ds.coords[rotvar] = ds_rot.coords[rotvar]
                scan_dims.append(rotvar)
        except NameError:
            pass
if fake_rotvar:
    ds.coords[rotvar] = np.linspace(0, 1, 10)
    scan_dims.append(rotvar)

# Look if we have a gam network somewhere deeper
gam_leq_nns = {nn_name: None for nn_name in nns.keys()}
for nn_name, nn in nns.items():
    if nn is not None and 'gam_leq_GB' not in nn._target_names:
        if hasattr(nn, '_internal_network'):
            if 'gam_leq_GB' in nn._internal_network._target_names.values:
                gam_leq_nns[nn_name] = nn._internal_network
            else:
                if hasattr(nn._internal_network, '_internal_network'):
                    if 'gam_leq_GB' in nn._internal_network._internal_network._target_names.values:
                        gam_leq_nns[nn_name] = nn._internal_network._internal_network
try:
    del nn
except NameError:
    pass


############################################################
# Create sliders                                           #
############################################################
round = CustomJS(code="""
         var f = cb_obj
         f = Number(f.toPrecision(2))
         return f
     """)
slider_dict = OrderedDict()
for name in scan_dims:
    # By default, color the slider green and put marker halfway
    start = float(ds[name].isel(**{name: int(ds[name].size/2)}))
    #start = int(ds[name].size/2)
    color = 'green'
    if name == rotvar:
        start = float(ds[name].isel({rotvar: np.abs(ds[name]).argmin()}))
    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name].data).tolist(),
                                       prefix=name + " = ",
                                       height=56,
                                       #prefix=name + " = ", height=50,
                                       value=(start, start),
                                       prettify=round,
                                       bar_color=color, title='', show_value=False)
# Link update event to all sliders
for slider in slider_dict.values():
    slider.on_change('value', updater)

# Display the sliders in two columns
height_block = 300
slidercol1 = widgetbox(list(slider_dict.values())[:len(slider_dict)//2],
                       height=int(.75*height_block),
                       name='slidercol1',
                       sizing_mode='scale_both'
                       )
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:],
                       height=int(.75*height_block),
                       name='slidercol2',
                       sizing_mode='scale_both'
                       )
sliderrow = row(slidercol1, slidercol2,
                sizing_mode='scale_both'
                )

# Create slider to select x-axis
xaxis_name = scan_dims[1]
start = xaxis_name
xaxis_slider = IonRangeSlider(values=scan_dims, height=56, value=(start, start), title='', show_value=False, name='xaxis_slider')
xaxis_slider.on_change('value', swap_x)
print_slice_button = Button(label="Print slice", button_type="success", name='print_slice_button')
print_slice_button.on_click(print_slice)
print_slice_text = Div(text="", name='print_slice_text')

toolbar = column([xaxis_slider, print_slice_button, print_slice_text], sizing_mode='scale_width')

############################################################
# Create figures                                           #
############################################################
flux_tools = ['box_zoom,pan,zoom_in,zoom_out,reset,save']

x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
sel_dict = read_sliders()
df_flux, df_freq, df_nn, df_rot  = extract_plotdata(sel_dict)
flux_source = ColumnDataSource(df_flux)
rot_source = ColumnDataSource({name: [] for name in df_flux})
freq_source = ColumnDataSource(df_freq)
nn_source = ColumnDataSource(df_nn)
# Define the flux-like plots (e.g. xaxis_name on the x-axis)
labels = {
    'ef': 'Energy flux',
    'pf': 'Particle flux',
    'gam': 'Growth rate',
    'df': 'Particle diffusivity',
    'vt': 'Particle thermodiffusion',
    'vc': 'Particle convection'
          }
sizing_mode = 'scale_both'
fluxfigs = OrderedDict()
for (pre, species, suffix, norm) in flux_vars:
    figname = pre + suffix
    if figname not in fluxfigs:
        tags = ['fluxlike', pre, species, norm]
        if suffix == '':
            tags.append('total')
        else:
            tags.append(suffix)
        fluxfigs[figname]   = Figure(x_axis_label=xaxis_name,
                                          y_axis_label=labels[pre] + ' ' + suffix + ' [' + norm[1:] + ']',
                                          height=2*height_block, width=2*height_block,
                                          tools=flux_tools, x_range=x_range, tags=tags,
                                          sizing_mode=sizing_mode
                                          )
        curdoc().add_root(fluxfigs[figname])

for fig in fluxfigs.values():
    hover = HoverTool()
    hover.tooltips = [('x,y', '($x, $y)')]
    fig.add_tools(hover)
plotrow_figs = list(fluxfigs.values())

# Define the frequency-like plots (e.g. kthetarhos at the x-axis)
figs = OrderedDict()
if plot_freq:
    freq_tools = []
    kthetarhos_cutoff = 1
    kthetarhos_cutoff_index = int(np.argwhere(
        np.isclose(ds['kthetarhos'].data, kthetarhos_cutoff)))
    figs['gamlow']  = Figure(x_axis_label=' ',
                             y_axis_label='Growth Rates [GB]',
                             #height=height_block, width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff],
                             toolbar_location=None, name='gamlow',
                             sizing_mode=sizing_mode,
                             )
    figs['gamhigh'] = Figure(x_axis_label=' ',
                             y_axis_label=' ',
                             #height=height_block, width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())],
                             toolbar_location=None, name='gamhigh',
                             sizing_mode=sizing_mode,
                             )
    figs['omelow']  = Figure(x_axis_label='kthetarhos',
                             y_axis_label='Frequencies [GB]',
                             #height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff],
                             toolbar_location=None, name='omelow',
                             sizing_mode=sizing_mode,
                             )
    figs['omehigh'] = Figure(x_axis_label='kthetarhos',
                             y_axis_label=' ',
                             #height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())],
                             toolbar_location=None, name='omehigh',
                             sizing_mode=sizing_mode,
                             )

############################################################
# Create legend, style and data sources for fluxplots      #
############################################################
from bokeh.palettes import Category20
sepcolor = Category20[20]
particle_names = ['e'] + ['i']
if plot_nn:
    nn_names = nns.keys()
else:
    nn_names = []
colors = {}
rot_colors = {}
for ii, species in enumerate(particle_names):
    rot_colors[species] = sepcolor[2*ii+1]
    colors[species] = sepcolor[2*ii]

style_dash = ['solid', 'dashed', 'dotted', 'dashdot', 'dotdash']
dashes = {'qlk': 'scatter'}
for ii, name in enumerate(nn_names):
    dashes[name] = style_dash[ii]

colors['_leq'] = sepcolor[-1]

# link data sources to figures
for pre, species, suff, norm in flux_vars:
    fluxname = ''.join([pre, species, suff, norm])
    figname = pre + suff
    fig = fluxfigs[figname]
    for nn_name in dashes.keys():
        if nn_name == 'qlk':
            if show_legend:
                legend = species
            else:
                legend = None
            glyph = fig.scatter('xaxis', fluxname,
                                source=flux_source,
                                color=colors[species],
                                legend=legend,
                               )
            glyph = fig.scatter('xaxis', fluxname,
                                source=rot_source,
                                color=rot_colors[species],
                                legend=legend,
                               )
        else:
            for nn in nns.values():
                if (fluxname in list(nn._target_names) or
                   (pre == 'gam' and not all([nn is None for nn in gam_leq_nns.values()]))):
                    if show_legend:
                        legend = nn_name
                    else:
                        legend = None
                    glyph = fig.line('xaxis', nn_name + '_' + fluxname,
                                     source=nn_source,
                                     color=colors[species],
                                     line_dash=dashes[nn_name],
                                     legend=legend,
                                    )
                    if plot_nn_eb:
                        band = Band(base='xaxis',
                                    lower=nn_name + '_' + fluxname + '_LEB',
                                    upper=nn_name + '_' + fluxname + '_UEB',
                                    source=nn_source,
                                    level='underlay',
                                    fill_alpha=1.0, line_width=0,
                                    line_color=colors[species])
                        fig.add_layout(band)
            del nn

############################################################
# Create legend, style and data sources for freqplots      #
############################################################
# Find the maximum size of dims to define plot colors
if plot_freq:
    max_dim_size = max([ds.dims[dim] for dim in scan_dims])
    num_kr = len(df_freq['kthetarhos'].unique())
    num_kr_leq = len(df_freq['kthetarhos'].loc[df_freq['kthetarhos'] <= kthetarhos_cutoff].unique())
    kr_leq = list(range(num_kr_leq))
    num_kr_great = len(df_freq['kthetarhos'].loc[df_freq['kthetarhos'] > kthetarhos_cutoff].unique())
    kr_great = list(range(num_kr_leq, num_kr_leq + num_kr_great))
    numsols = len(df_freq['numsols'].unique())
    numsol_filter = OrderedDict()
    for sol in range(numsols):
        numsol_filter[sol] = set(range(sol, max_dim_size * num_kr * numsols + sol, numsols))
    kr_filter = OrderedDict()
    for kr in range(num_kr):
        kr_filter[kr] = set()
        for dim in range(max_dim_size):
            kr_filter[kr] |= set(range(numsols * kr + dim * numsols * num_kr , numsols * (kr + 1) + dim * numsols * num_kr))

    dim_filter = OrderedDict()
    for dim in range(max_dim_size):
        dim_filter[dim] = set(range(num_kr * numsols * dim, num_kr * numsols * (dim + 1)))

    from bokeh.models import CDSView, IndexFilter, BooleanFilter, GroupFilter
    renderers = []
    views = []
    lines = []
    for sol in range(numsols):
        for dim in range(max_dim_size):
            idx = dim_filter[dim] & numsol_filter[sol]
            for krs in [kr_leq, kr_great]:
                kr_filters = [kr_filter[kr] & idx for kr in krs]
                idx_kr = sorted(set.union(*kr_filters))
                filt = IndexFilter(idx_kr)
                view = CDSView(source=freq_source, filters=[filt])
                views.append(view)
                #opts = dict(source=freq_source, color=color, alpha=.5,
                #            name=column_name, hover_color=color)
                opts = dict(source=freq_source, alpha=.5, view=view)
                if max(krs) < num_kr_leq: #leq
                    kr = 'low'
                else: #great
                    kr = 'high'
                for name in ['ome', 'gam']:
                    figs[name + kr].scatter('kthetarhos', name + norm, **opts)
                    line = figs[name + kr].line('kthetarhos', name + norm, **opts)
                    lines.append(line)


        #seqcolor = takespread(Plasma256, max_dim_size,
        #                      repeat=int(ds.dims['numsols']))
        #for color, column_name in zip(seqcolor, linenames):
        #    figs[figname].scatter('xaxis', 'yaxis', **opts)
        #    figs[figname].line('xaxis', 'yaxis', **opts)
    for kr, name in product(['low', 'high'], ['gam', 'ome']):
        figs[name + kr].add_tools(HoverTool(tooltips=OrderedDict([('sol', '@numsols')])))
        #                                                        ('sol', '@cursol')]
        #                                                      )))

from itertools import chain
if __name__ == '__main__':
    embed()
else:
    for fig in chain(fluxfigs.values(), figs.values()):
        curdoc().add_root(fig)
    curdoc().add_root(slidercol1)
    curdoc().add_root(slidercol2)
    curdoc().add_root(xaxis_slider)
    curdoc().add_root(print_slice_button)
    curdoc().add_root(print_slice_text)
