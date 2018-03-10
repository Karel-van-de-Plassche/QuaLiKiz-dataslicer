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
from bokeh_ion_range_slider.ionrangeslider import IonRangeSlider
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
try:
    from qlknn.models.ffnn import QuaLiKizNDNN
except ModuleNotFoundError:
    plot_nn = False
else:
    try:
        import mega_nn
    except:
        print('No mega NN')
    plot_nn = True

def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]

def extract_plotdata(sel_dict):
    start = time.time()
    slice_ = ds.sel(**sel_dict)
    #slice_sep = ds_sep.sel(**sel_dict)
    slice_grow = ds_grow.sel(**sel_dict)
    slice_ = xr.merge([slice_, slice_grow])
    #slice_ = slice_.sel(nions=0)
    slice_.load()

    #slice_ = slice_.where(slice_['efe_GB'] < 60)
    #slice_ = slice_.where(slice_['efi_GB'] < 60)
    #slice_ = slice_.where(slice_['efe_GB'] > 0.1)
    #slice_ = slice_.where(slice_['efi_GB'] > 0.1)
    xaxis = slice_[xaxis_name].data
    plotdata = {}
    for prefix in ['ef', 'pf', 'df', 'pinch', 'grow']:
        plotdata[prefix + 'fig'] = {}
    for prefix in ['ef', 'pf', 'df', 'pinch']:
        for suffix in flux_suffixes:
            plotdata[prefix + 'fig' + suffix] = {}

    if plot_nn:
        nn_xaxis = np.linspace(xaxis[0], xaxis[-1], 60)
        input = pd.DataFrame({xaxis_name: nn_xaxis})
        for name in nn._feature_names:
            #input = df[[x for x in nn.feature_names if x != xaxis_name]].groupby(level=0).max().reset_index()
            if name != xaxis_name:
                if name == 'logNustar':
                    val = np.log10(slice_['Nustar'])
                else:
                    val = slice_[name]
                input[name] = np.full_like(nn_xaxis, val)
        output = nn.get_output(input)
        for name in ['efe_GB', 'efi_GB', 'pfe_GB']:
            try:
                output[name]
            except KeyError:
                output[name] = np.zeros_like(output.index)


        #timer('nn eval at ', start)
        if plot_ef:
            prefix = 'ef'
            for suffix in flux_suffixes:
                for species in ['elec', 'ion0']:
                    for nn_suffix in ['', '_A', '_B', '_C']:
                        plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species] = {}
                        try:
                            plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species]['xaxis'] = nn_xaxis
                            plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species]['yaxis'] = output[prefix + species[0] + suffix[1:] + '_GB' + nn_suffix]
                        except KeyError:
                            pass
        if plot_pf:
            prefix = 'pf'
            for suffix in flux_suffixes:
                #for species in ['elec', 'ion0']:
                for species in ['elec']:
                    for nn_suffix in ['', '_A', '_B', '_C']:
                        plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species] = {}
                        try:
                            plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species]['xaxis'] = nn_xaxis
                            plotdata[prefix + 'fig' + suffix]['nn' + nn_suffix[1:] + '_' + species]['yaxis'] = output[prefix + species[0] + suffix[1:] + '_GB' + nn_suffix]
                        except KeyError:
                            pass
        if plot_df:
            prefix = 'df'
            plotdata[prefix + 'fig']['nn_elec'] = {}
            plotdata[prefix + 'fig']['nn_elec']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn_elec']['yaxis'] = output[prefix + 'e_GB']
            plotdata[prefix + 'fig']['nn_ion0'] = {}
            plotdata[prefix + 'fig']['nn_ion0']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn_ion0']['yaxis'] = output[prefix + 'i_GB']
        if plot_pinch:
            prefix = 'pinch'
            plotdata[prefix + 'fig']['nn_elec'] = {}
            plotdata[prefix + 'fig']['nn_elec']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn_elec']['yaxis'] = output['vte_GB_plus_vce_GB']
            plotdata[prefix + 'fig']['nn_ion0'] = {}
            plotdata[prefix + 'fig']['nn_ion0']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn_ion0']['yaxis'] = output['vti_GB_plus_vci_GB']
            plotdata[prefix + 'fig']['nn2_elec'] = {}
            plotdata[prefix + 'fig']['nn2_elec']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn2_elec']['yaxis'] = output['vte_GB_plus_vce_GB2']
            plotdata[prefix + 'fig']['nn2_ion0'] = {}
            plotdata[prefix + 'fig']['nn2_ion0']['xaxis'] = nn_xaxis
            plotdata[prefix + 'fig']['nn2_ion0']['yaxis'] = output['vti_GB_plus_vci_GB2']
        #if plot_grow:
        #    prefix = 'grow'
        #    plotdata[prefix + 'fig']['nn_leq'] = {}
        #    plotdata[prefix + 'fig']['nn_leq']['xaxis'] = nn_xaxis
        #    plotdata[prefix + 'fig']['nn_leq']['yaxis'] = output['gam_GB_leq2max']
        #    plotdata[prefix + 'fig']['nn_less'] = {}
        #    plotdata[prefix + 'fig']['nn_less']['xaxis'] = nn_xaxis
        #    plotdata[prefix + 'fig']['nn_less']['yaxis'] = output['gam_GB_less2max']

        #timer('nn dictized at ', start)
    for prefix in ['ef', 'pf', 'df']:
        for suffix in flux_suffixes:
            if prefix + 'fig' in figs:
                for species in ['elec', 'ion0']:
                    if prefix == 'pf' and species == 'ion0':
                        continue
                    try:
                        plotdata[prefix + 'fig' + suffix][species] = {}
                        plotdata[prefix + 'fig' + suffix][species]['xaxis'] = xaxis
                        plotdata[prefix + 'fig' + suffix][species]['yaxis'] = slice_[prefix + species[0] + suffix[1:] + '_GB'].data
                    except KeyError:
                        pass

    if plot_grow:
        prefix = 'grow'
        plotdata[prefix + 'fig']['leq'] = {}
        plotdata[prefix + 'fig']['leq']['xaxis'] = xaxis
        plotdata[prefix + 'fig']['leq']['yaxis'] = slice_['gam_GB'].where(slice_['kthetarhos'] >= 2).max(['kthetarhos', 'numsols']).data
        plotdata[prefix + 'fig']['less'] = {}
        plotdata[prefix + 'fig']['less']['xaxis'] = xaxis
        plotdata[prefix + 'fig']['less']['yaxis'] = slice_['gam_GB'].where(slice_['kthetarhos'] < 2).max(['kthetarhos', 'numsols']).data

    if plot_pinch:
        prefix = 'pinch'
        for ii, (vti, vci) in enumerate(zip(slice_['vti_GB'].T, slice_['vci_GB'].T)):
            if ii > 0:
                break
            plotdata[prefix + 'fig']['ion' + str(ii)] = {}
            plotdata[prefix + 'fig']['ion' + str(ii)]['xaxis'] = xaxis
            plotdata[prefix + 'fig']['ion' + str(ii)]['yaxis'] = vti.data + vci.data
        plotdata[prefix + 'fig']['elec'] = {}
        plotdata[prefix + 'fig']['elec']['xaxis'] = xaxis
        plotdata[prefix + 'fig']['elec']['yaxis'] = slice_['vte_GB'].data + slice_['vte_GB'].data
    #timer('fluxed at ', start)

    #if all([fig in figs for fig in ['gamlow', 'gamhigh', 'omelow', 'omehigh']]):
    if plot_freq:
        for suff in ['low', 'high']:
            for pre in ['gam', 'ome']:
                plotdata[pre + suff] = {}
        for numsol, __ in enumerate(slice_['numsols'].data):
            for suff in ['low', 'high']:
                for pre in ['gam', 'ome']:
                    if suff == 'low':
                        kthetarhos = slice(None, kthetarhos_cutoff_index)
                    else:
                        kthetarhos = slice(kthetarhos_cutoff_index, None)
                    subslice = slice_[pre + '_GB'].isel(numsols=numsol, kthetarhos=kthetarhos)
                    subslice = subslice.where(subslice != 0)

                    for ii, subsubslice in enumerate(subslice):
                        plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))] = {}
                        plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['xaxis'] = subslice['kthetarhos'].data
                        plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['yaxis'] = subsubslice.data
                        plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['curval'] = np.full_like(subslice['kthetarhos'], subsubslice[xaxis_name].data)
                        plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['cursol'] = np.full_like(subslice['kthetarhos'], int(numsol))
    #timer('freq at ', start)
    return plotdata


def swap_x(attr, old, new):
    global xaxis_name, sources, figs
    try:
        xaxis_name = xaxis_slider.values[new[0]]
        old_xaxis_name = xaxis_slider.values[old[0]]
    except TypeError:
        return

    if plot_freq:
        for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
            for column_name in sources[figname]:
                sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}

    for figpre in ['effig', 'pffig', 'dffig', 'pinchfig', 'growfig']:
        for suffix in flux_suffixes:
            figname = figpre + suffix
            if figname in figs:
                for column_name in sources[figname]:
                    sources[figname][column_name].data = {'xaxis': [], 'yaxis': []}

                figs[figname].x_range.start = float(np.min(ds[xaxis_name]))
                figs[figname].x_range.end = float(np.max(ds[xaxis_name]))
                figs[figname].xaxis.axis_label = xaxis_name
    updater(None, None, None)

def read_sliders():
    sel_dict = {}
    for name, slider in slider_dict.items():
        if name != xaxis_name:
            sel_dict[name] = slider.values[slider.range[0]]
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
    #timer('Read sliders ', start)

    plotdata = extract_plotdata(sel_dict)
    #timer('Extracted plotdata', start)
    for figpre in ['effig', 'pffig', 'dffig', 'pinchfig', 'growfig']:
        for suffix in flux_suffixes:
            figname = figpre + suffix
            if figname in figs:
                for column_name in plotdata[figname]:
                    sources[figname][column_name].data = plotdata[figname][column_name]
    #timer('wrote flux sources', start)
    if plot_freq:
        for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
            for column_name in sources[figname]:
                if column_name in plotdata[figname]:
                    sources[figname][column_name].data = plotdata[figname][column_name]
                else:
                    sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}
    #timer('wrote freq sources', start)
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

#ds_sep = xr.open_dataset(os.path.join(root_dir, 'Zeffcombo.sep.nc.1'))
#ds_sep = ds_sep.drop([x for x in ds_sep.coords if x not in ds_sep.dims and x not in ['Zi']])
#ds = xr.open_dataset('4D.nc3')

plot_nn = plot_nn and True
plot_freq = True
plot_ef = True
plot_pf = False
plot_pinch = False
plot_df = False
plot_grow = False
plot_sepflux = True

sepflux_names = ['ETG', 'ITG', 'TEM']
#sepflux_names = ['ETG']
if plot_sepflux:
    flux_suffixes = [''] + ['_' + name for name in sepflux_names]
else:
    flux_suffixes = ['']

if plot_nn:
    #nn = QuaLiKizNDNN.from_json('nn.json')
    nn = mega_nn.nn

scan_dims = [name for name in ds.dims if name not in ['nions', 'numsols', 'kthetarhos']]

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
    start = int(ds[name].size/2)
    color = 'green'
    if plot_nn:
        try:
            if nn._feature_min[name] == nn._feature_max[name]:
                # If there are features with a specific value, color the bar red
                # and put marker on value
                start = int(np.argwhere(
                    np.isclose(ds[name], nn._feature_min[name], rtol=1e-2)))
                color = 'red'
        except KeyError:
            print('No ' + name + ' value found')

    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name].data).tolist(),
                                       prefix=name + " = ", height=56,
                                       #prefix=name + " = ", height=50,
                                       prettify=round, start=start,
                                       color=color)
# Link update event to all sliders
for slider in slider_dict.values():
    slider.on_change('range', updater)

# Display the sliders in two columns
height_block = 300
slidercol1 = widgetbox(list(slider_dict.values())[:len(slider_dict)//2],
                       height=int(.75*height_block))
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:],
                       height=int(.75*height_block))
sliderrow = row(slidercol1, slidercol2, sizing_mode='scale_width')

# Create slider to select x-axis
xaxis_name = scan_dims[1]
xaxis_slider = IonRangeSlider(values=scan_dims, height=56, start=scan_dims.index(xaxis_name))
xaxis_slider.on_change('range', swap_x)
print_slice_button = Button(label="Print slice", button_type="success")
print_slice_button.on_click(print_slice)
print_slice_text = Div(text="")

toolbar = column([xaxis_slider, print_slice_button, print_slice_text], sizing_mode='scale_width')

############################################################
# Create figures                                           #
############################################################
flux_tools = ['box_zoom,pan,zoom_in,zoom_out,reset,save']

x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
figs = OrderedDict()
# Define the flux-like plots (e.g. xaxis_name on the x-axis)
if plot_ef:
        for suffix in flux_suffixes:
            figs['effig' + suffix]   = Figure(x_axis_label=xaxis_name,
                                              y_axis_label='Energy Flux ' + suffix[1:] + ' [GB]',
                                              height=2*height_block, width=2*height_block,
                                              tools=flux_tools, x_range=x_range)

if plot_pf:
    for suffix in flux_suffixes:
        figs['pffig' + suffix]   = Figure(x_axis_label=xaxis_name,
                                          y_axis_label='Particle Flux ' + suffix[1:] + ' [GB]',
                                          height=2*height_block, width=2*height_block,
                                          tools=flux_tools, x_range=x_range)
if plot_df:
    figs['dffig']   = Figure(x_axis_label=xaxis_name,
                             y_axis_label='Particle Diffusion [GB]',
                             height=2*height_block, width=2*height_block,
                             tools=flux_tools, x_range=x_range)
if plot_pinch:
    figs['pinchfig']   = Figure(x_axis_label=xaxis_name,
                             y_axis_label='Particle Pinch [GB]',
                             height=2*height_block, width=2*height_block,
                             tools=flux_tools, x_range=x_range)
if plot_grow:
    figs['growfig']   = Figure(x_axis_label=xaxis_name,
                             y_axis_label='Maximum Growth Rate [GB]',
                             height=2*height_block, width=2*height_block,
                             tools=flux_tools, x_range=x_range)

for fig in figs.values():
    hover = HoverTool()
    hover.tooltips = [('x,y', '(@xaxis, @yaxis)')]
    fig.add_tools(hover)
plotrow_figs = list(figs.values())

# Define the frequency-like plots (e.g. kthetarhos at the x-axis)
if plot_freq:
    freq_tools = 'save'
    kthetarhos_cutoff = 1
    kthetarhos_cutoff_index = int(np.argwhere(
        np.isclose(ds['kthetarhos'].data, kthetarhos_cutoff)))
    figs['gamlow']  = Figure(x_axis_label=' ',
                             y_axis_label='Growth Rates [GB]',
                             height=height_block, width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff])
    figs['gamhigh'] = Figure(x_axis_label=' ',
                             y_axis_label=' ',
                             height=height_block, width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())])
    figs['omelow']  = Figure(x_axis_label='kthetarhos',
                             y_axis_label='Frequencies [GB]',
                             height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[0, kthetarhos_cutoff])
    figs['omehigh'] = Figure(x_axis_label='kthetarhos',
                             y_axis_label=' ',
                             height=height_block,   width=height_block,
                             tools=freq_tools, x_range=[kthetarhos_cutoff,
                                                        float(ds['kthetarhos'].max())])
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
particle_names = ['elec'] + ['Z = 1']

                             #+ str(Zi.data) for Zi in ds['Zi']]
style_names = ['', 'nn_', 'nnA_', 'nnB_', 'nnC_']
style_dash = ['solid', 'dashed', 'dotted', 'dashdot', 'dotdash']
lines = OrderedDict()
for (num_style, style), (num_part, part) in product(enumerate(style_names), enumerate(particle_names)):
    if num_part > 1:
        continue
    if part == 'elec':
        name = 'elec'
    else:
        name = 'ion' + str(num_part - 1)
    name = style + name
    line = lines[name] = {}
    line['color'] = sepcolor[num_part]
    line['dash'] = style_dash[num_style]
    line['legend'] = style + part

# link data sources to figures
legend_added = False
sources = {}
for fluxname in ['effig', 'pffig', 'pinchfig', 'dffig']:
    for flux_suffix in flux_suffixes:
        figname = fluxname + flux_suffix
        if figname in figs:
            sources[figname] = OrderedDict()
            fig = figs[figname]
            for line_name, line in lines.items():
                source = sources[figname][line_name] = ColumnDataSource({'xaxis': [],
                                                                         'yaxis': []})
                if 'nn' in line_name:
                    if plot_nn:

                        glyph = fig.line('xaxis', 'yaxis',
                                           source=source,
                                           color=line['color'],
                                           #legend=legend[column_name],
                                           line_dash=line['dash'])
                else:
                    glyph = fig.scatter('xaxis', 'yaxis',
                                          source=source,
                                          color=line['color'],
                                          #legend=legend[column_name],
                                          size=6)
                line['glyph'] = glyph
            #figs[figname].legend.location = 'top_left'
            #figs[figname].legend[0].items.clear()
            #if not legend_added:
            #    legends = [(line['legend'], [line['glyph']]) for line in lines.values()]
            #    legend = Legend(legends=legends, location=(0,0), orientation='horizontal')
            #    fig.add_layout(legend, 'above')
            #    legend_added = True
############################################################
# Create legend, style and data sources for growplots      #
###########################################################
color = OrderedDict([('less', sepcolor[-1]),
                     ('leq', sepcolor[-2]),
                     ('nn_less', sepcolor[-1]),
                     ('nn_leq', sepcolor[-2])])
line_dash = OrderedDict([('less', 'solid'),
                         ('leq', 'solid'),
                         ('nn_less', 'dashed'),
                         ('nn_leq', 'dashed')])
legend = OrderedDict([('less', 'kr < 2'),
                      ('leq', 'kr >= 2'),
                      ('nn_less', 'nn_kr < 2'),
                      ('nn_leq', 'nn_kr >= 2')])
linenames = ['less', 'leq']
if plot_nn:
    linenames.extend(['nn_less', 'nn_leq'])

# link data sources to figures
for figname in ['growfig']:
    if figname in figs:
        sources[figname] = OrderedDict()
        for ii, column_name in enumerate(linenames):
            sources[figname][column_name] = ColumnDataSource({'xaxis': [],
                                                              'yaxis': []})
            if 'nn' in column_name:
                if plot_nn:
                    figs[figname].line('xaxis', 'yaxis',
                                       source=sources[figname][column_name],
                                       color=color[column_name],
                                       legend=legend[column_name],
                                       line_dash=line_dash[column_name])
            else:
                figs[figname].scatter('xaxis', 'yaxis',
                                      source=sources[figname][column_name],
                                      color=color[column_name],
                                      legend=legend[column_name],
                                      size=6)
        figs[figname].legend.location = 'top_left'

############################################################
# Create legend, style and data sources for freqplots      #
############################################################
# Find the maximum size of dims to define plot colors
if plot_freq:
    max_dim_size = 0
    for scan_dim in scan_dims:
        num = ds.dims[scan_dim]
        if num > max_dim_size:
            max_dim_size = num

    linenames = []
    for ii in range(max_dim_size):
        for jj in range(ds.dims['numsols']):
            linenames.append('dim' + str(ii) + 'sol' + str(jj))

    renderers = []
    for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
        seqcolor = takespread(Plasma256, max_dim_size,
                              repeat=int(ds.dims['numsols']))
        sources[figname] = OrderedDict()
        for color, column_name in zip(seqcolor, linenames):
            source = sources[figname][column_name] = ColumnDataSource({'xaxis': [],
                                                                       'yaxis': [],
                                                                       'curval': [],
                                                                       'cursol': []})
            opts = dict(source=source, color=color, alpha=.5,
                        name=column_name, hover_color=color)
            figs[figname].scatter('xaxis', 'yaxis', **opts)
            figs[figname].line('xaxis', 'yaxis', **opts)
        figs[figname].add_tools(HoverTool(tooltips=OrderedDict([('val', '@curval'),
                                                                ('sol', '@cursol')]
                                                              )))

layout = column(plotrow, sliderrow, toolbar, sizing_mode='scale_width')
if __name__ == '__main__':
    embed()
    #show(layout)
else:
    curdoc().add_root(layout)
