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
from bokeh_ion_rangeslider import IonRangeSlider
from bokeh.plotting import figure, show, reset_output, Figure
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox
from bokeh.models.widgets import Button, Div
from bokeh.models import HoverTool
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, CustomJS, Legend, Line, Circle
from bokeh.palettes import Set1 as sepcolor
from bokeh.palettes import Plasma256
#TODO: Add sane checking if loading failed and why
import sys
sys.path.append('../QLKNN-develop')
try:
    ModuleNotFoundError
except:
    ModuleNotFoundError = ImportError
plot_nn = False
try:
    from qlknn.models.ffnn import QuaLiKizNDNN
except ModuleNotFoundError:
    pass
else:
    try:
        import mega_nn
        plot_nn = True
    except:
        print('No mega NN')

def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]

def extract_plotdata(sel_dict):
    start = time.time()
    slice_ = ds.sel(**sel_dict)
    slice_.load()
    #slice_sep = ds_sep.sel(**sel_dict)
    #slice_grow = ds_grow.sel(**sel_dict)
    #slice_ = xr.merge([slice_, slice_grow])
    #slice_ = slice_.sel(nions=0)
    df_flux = slice_[fluxlike_vars].reset_coords(drop=True).to_dataframe()
    df_flux.index.name = 'xaxis'
    df_flux.reset_index(inplace=True)

    if plot_nn:
        xaxis = df_flux['xaxis']
        nn_xaxis = np.linspace(xaxis.iloc[0], xaxis.iloc[-1], 60)

        inp = pd.DataFrame({name: float(slice_[name]) for name in nn._feature_names if name != xaxis_name and name in slice_}, index=[0])
        input = pd.DataFrame({xaxis_name: np.linspace(xaxis.iloc[0], xaxis.iloc[-1], 60)}).join(inp).fillna(method='ffill')
        df_nn = nn.get_output(input)
        df_nn.drop([name for name in nn._target_names if not name in fluxlike_vars], axis=1, inplace=True)
        df_nn.columns = ['nn_' + name for name in df_nn.columns]
        df_nn.index = (input[xaxis_name])
        df_nn.index.name = 'xaxis'
        df_nn.reset_index(inplace=True)
    else:
        df_nn = pd.DataFrame()

    if plot_freq:
        df_freq = slice_[freq_vars].reset_coords(drop=True).to_dataframe()
        df_freq.reset_index(inplace=True)
    else:
        df_freq = pd.DataFrame()
    #df_freq.index.set_names('xaxis', level='kthetarhos', inplace=True)

    if plot_victor and slice_['gammaE'] != 0:
        df_flux = pd.DataFrame()

    return df_flux, df_freq, df_nn

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

    df_flux, df_grow, df_nn = extract_plotdata(sel_dict)
    for source, df in zip([flux_source, freq_source, nn_source], [df_flux, df_freq, df_nn]):
        if len(df) == 0:
            source.data = {name: [] for name in source.column_names}
        else:
            source.data = dict(df)

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


############################################################
# Load dataset                                             #
############################################################
import socket
if socket.gethostname().startswith('rs'):
    root_dir = '/Rijnh/Shares/Departments/Fusiefysica/IMT/karel'
else:
    root_dir = '../qlk_data'
ds = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.combo.nions0.nc.1'))
#ds = xr.open_dataset(os.path.join(root, 'Zeffcombo.nc.1'))
ds = ds.drop([x for x in ds.coords if x not in ds.dims and x not in ['Zi']])
ds_grow = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.grow.nc'))
ds_grow = ds_grow.drop([x for x in ds_grow.coords if x not in ds_grow.dims and x not in ['Zi']])
ds = ds.merge(ds_grow)
ds['logNustar'] = np.log10(ds['Nustar'])
ds = ds.swap_dims({'Nustar': 'logNustar'})

#ds_sep = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.sep.nc.1'))
#ds_sep = ds_sep.drop([x for x in ds_sep.coords if x not in ds_sep.dims and x not in ['Zi']])
#ds = xr.open_dataset('4D.nc3')

plot_nn = plot_nn and True
plot_freq = False
plot_ef = True
plot_pf = False
plot_pinch = False
plot_df = False
plot_grow = False
plot_sepflux = True
plot_victor = True
norm = '_GB'

sepflux_names = ['ETG', 'ITG', 'TEM']
#sepflux_names = ['ETG']
if plot_sepflux:
    flux_suffixes = [''] + [name for name in sepflux_names]
else:
    flux_suffixes = ['']

if plot_nn:
    #nn = QuaLiKizNDNN.from_json('nn.json')
    nn = mega_nn.nn
else:
    nn = None

scan_dims = [name for name in ds.dims if name not in ['nions', 'numsols', 'kthetarhos']]
flux_vars = []
for pre in ['ef', 'pf']:
    if ((pre == 'ef' and not plot_ef) or
        (pre == 'pf' and not plot_pf)):
        continue
    for suff in flux_suffixes:
        for species in ['i', 'e']:
            if ((species == 'i' and suff == 'ETG') or
                (pre == 'pf' and species == 'e')):
                continue
            flux_vars.append((pre, species, suff, norm))
if plot_freq:
    freq_vars = [name for name in ds.data_vars if any(var in name for var in ['ome' + norm, 'gam' + norm])]
else:
    freq_vars = []

if plot_grow:
    grow_vars = [('gam_leq', '', '', norm)]
else:
    grow_vars = []
fluxlike_vars = [''.join(var) for var in flux_vars + grow_vars]
ds = ds.drop([name for name in ds.data_vars if name not in freq_vars + fluxlike_vars])

if plot_victor:
    ds.coords['gammaE'] = np.linspace(0, 1, 10)
    scan_dims.append('gammaE')

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
    if name == 'gammaE':
        start = 0
    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name].data).tolist(),
                                       prefix=name + " = ", height=56,
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
                       height=int(.75*height_block))
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:],
                       height=int(.75*height_block))
sliderrow = row(slidercol1, slidercol2, sizing_mode='scale_width')

# Create slider to select x-axis
xaxis_name = scan_dims[1]
start = xaxis_name
xaxis_slider = IonRangeSlider(values=scan_dims, height=56, value=(start, start), title='', show_value=False)
xaxis_slider.on_change('value', swap_x)
print_slice_button = Button(label="Print slice", button_type="success")
print_slice_button.on_click(print_slice)
print_slice_text = Div(text="")

toolbar = column([xaxis_slider, print_slice_button, print_slice_text], sizing_mode='scale_width')

############################################################
# Create figures                                           #
############################################################
flux_tools = ['box_zoom,pan,zoom_in,zoom_out,reset,save']

x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
sel_dict = read_sliders()
df_flux, df_freq, df_nn  = extract_plotdata(sel_dict)
flux_source = ColumnDataSource(df_flux)
freq_source = ColumnDataSource(df_freq)
nn_source = ColumnDataSource(df_nn)
fluxfigs = OrderedDict()
# Define the flux-like plots (e.g. xaxis_name on the x-axis)
labels = {
    'ef': 'Energy Flux',
    'pf': 'Particle flux',
    'grow': 'Growth rate'
          }
for figname in ['ef', 'pf', 'grow']:
    if ((figname == 'ef' and not plot_ef) or
        (figname == 'pf' and not plot_pf) or
        (figname == 'grow' and not plot_grow)):
            continue
    for suffix in flux_suffixes:
        if figname == 'pf' and suffix == 'ETG':
            continue
        if figname == 'grow' and suffix != '':
            continue
        fluxfigs[figname + suffix]   = Figure(x_axis_label=xaxis_name,
                                          y_axis_label=labels[figname] + ' ' + suffix + ' [' + norm[1:] + ']',
                                          height=2*height_block, width=2*height_block,
                                          tools=flux_tools, x_range=x_range)

for fig in fluxfigs.values():
    hover = HoverTool()
    hover.tooltips = [('x,y', '(@xaxis, @yaxis)')]
    fig.add_tools(hover)
plotrow_figs = list(fluxfigs.values())

# Define the frequency-like plots (e.g. kthetarhos at the x-axis)
if plot_freq:
    freq_tools = []
    kthetarhos_cutoff = 1
    kthetarhos_cutoff_index = int(np.argwhere(
        np.isclose(ds['kthetarhos'].data, kthetarhos_cutoff)))
    figs['gamlow']  = Figure(x_axis_label=' ',
                             y_axis_label='Growth Rates [GB]',
                             height=height_block, width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff],
                             toolbar_location=None)
    figs['gamhigh'] = Figure(x_axis_label=' ',
                             y_axis_label=' ',
                             height=height_block, width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())],
                             toolbar_location=None)
    figs['omelow']  = Figure(x_axis_label='kthetarhos',
                             y_axis_label='Frequencies [GB]',
                             height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff],
                             toolbar_location=None)
    figs['omehigh'] = Figure(x_axis_label='kthetarhos',
                             y_axis_label=' ',
                             height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())],
                             toolbar_location=None)
    gamrow = row(figs['gamlow'], figs['gamhigh'],
                 height=height_block, width=height_block, sizing_mode='scale_width')
    omerow = row(figs['omelow'], figs['omehigh'],
                 height=height_block, width=height_block, sizing_mode='scale_width')
    freqgrid = column(gamrow, omerow, height=2*height_block, sizing_mode='scale_width')
    plotrow_figs.append(freqgrid)

plotrow = row(plotrow_figs,
              sizing_mode='scale_width', height=2*height_block)

############################################################
# Create legend, style and data sources for fluxplots      #
############################################################
sepcolor = sepcolor[9]
particle_names = ['e'] + ['i']
if nn:
    nn_names = ['nn_']
else:
    nn_names = []
colors = {}
for ii, species in enumerate(particle_names):
    colors[species] = sepcolor[ii]

style_dash = ['solid', 'dashed', 'dotted', 'dashdot', 'dotdash']
dashes = {'qlk': 'scatter'}
for ii, name in enumerate(nn_names):
    dashes[name] = style_dash[ii]

# link data sources to figures
for fluxname in flux_vars:
    pre, species, suff, norm = fluxname
    fig = fluxfigs[pre+suff]
    for linetype in dashes.keys():
        species_id = species[0]
        varname = pre + species_id + suff + norm
        if linetype == 'qlk':
            glyph = fig.scatter('xaxis', varname,
                                source=flux_source,
                                color=colors[species_id],
                                legend=species
                               )
        else:
            if varname in list(nn._target_names):
                glyph = fig.line('xaxis', linetype + varname,
                                 source=nn_source,
                                 color=colors[species_id],
                                 line_dash=dashes[name],
                                 legend=linetype
                                )

############################################################
# Create legend, style and data sources for growplots      #
###########################################################
colors = {'gam_leq': sepcolor[-1]}
# link data sources to figures
for fluxname in grow_vars:
    pre, species, suff, norm = fluxname
    fig = fluxfigs['grow']
    for linetype in dashes.keys():
        if linetype == 'qlk':
            glyph = fig.scatter('xaxis', pre + suff + norm,
                                source=flux_source,
                                color=colors[pre + suff],
                                legend=species
                               )
        else:
            glyph = fig.line('xaxis', linetype + pre + suff + norm,
                             source=nn_source,
                             color=colors[pre + suff],
                             line_dash=dashes[name],
                             legend=linetype
                             )

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

layout = column(plotrow, sliderrow, toolbar, sizing_mode='scale_width')
if __name__ == '__main__':
    embed()
    #show(layout)
else:
    curdoc().add_root(layout)
