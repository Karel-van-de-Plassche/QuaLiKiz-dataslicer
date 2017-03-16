"""
Copyright Dutch Institute for Fundamental Energy Research (2016)
Contributors: Karel van de Plassche (karelvandeplassche@gmail.com)
License: CeCILL v2.1
"""
import xarray as xr
import pandas as pd
import numpy as np
import scipy as sc
import os
import time
from collections import OrderedDict
from math import ceil
from itertools import repeat

from IPython import embed
from bokeh_ion_range_slider.ionrangeslider import IonRangeSlider
from bokeh.plotting import figure, show, reset_output, Figure
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox
from bokeh.models import HoverTool
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.palettes import Set1 as sepcolor
from bokeh.palettes import Plasma256
#from qlkANNk import QuaLiKiz4DNN
try:
    from run_model import QuaLiKizNDNN
    plot_nn = True
except ModuleNotFoundError:
    plot_nn = False


def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]

def extract_plotdata(sel_dict):
    start = time.time()
    slice_ = ds.sel(**sel_dict)
    slice_.load()

    slice_ = slice_.where(slice_['efe_GB'] < 60)
    xaxis = slice_[xaxis_name].data
    plotdata = {}
    for prefix in ['ef', 'pf']:
        plotdata[prefix + 'fig'] = {}

    if plot_nn:
        nn_xaxis = np.linspace(xaxis[0], xaxis[-1], 60)
        input = {xaxis_name: nn_xaxis}
        for name in nn.feature_names:
            #input = df[[x for x in nn.feature_names if x != xaxis_name]].groupby(level=0).max().reset_index()
            if name != xaxis_name:
                input[name] = np.full_like(nn_xaxis, slice_[name])
        output = nn.get_output(**input)
        for name in ['efe_GB', 'efi_GB', 'pfe_GB']:
            try:
                output[name]
            except KeyError:
                output[name] = np.zeros_like(output.index)

            
        #timer('nn eval at ', start)
        if plot_ef:
            plotdata['effig']['nn_elec'] = {}
            plotdata['effig']['nn_elec']['xaxis'] = nn_xaxis
            plotdata['effig']['nn_elec']['yaxis'] = output['efe_GB']
            plotdata['effig']['nn_ion0'] = {}
            plotdata['effig']['nn_ion0']['xaxis'] = nn_xaxis
            plotdata['effig']['nn_ion0']['yaxis'] = output['efi_GB']
        if plot_pf:
            plotdata['pffig']['nn_elec'] = {}
            plotdata['pffig']['nn_elec']['xaxis'] = nn_xaxis
            plotdata['pffig']['nn_elec']['yaxis'] = output['pfe_GB']

        #timer('nn dictized at ', start)
    for prefix in ['ef', 'pf']:
        if prefix + 'fig' in figs:
            for i, efi in enumerate(slice_[prefix + 'i_GB'].T):
                plotdata[prefix + 'fig']['ion' + str(i)] = {}
                plotdata[prefix + 'fig']['ion' + str(i)]['xaxis'] = xaxis
                plotdata[prefix + 'fig']['ion' + str(i)]['yaxis'] = efi.data
            plotdata[prefix + 'fig']['elec'] = {}
            plotdata[prefix + 'fig']['elec']['xaxis'] = xaxis
            plotdata[prefix + 'fig']['elec']['yaxis'] = slice_[prefix + 'e_GB'].data
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

    for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
        for column_name in sources[figname]:
            sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}

    for figname in ['effig', 'pffig']:
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
    for figname in ['effig', 'pffig']:
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
        if dim in nn.feature_names.values:
            if nn.feature_min[dim] != nn.feature_max[dim]:
                nn_scan_dims.append(dim)
    return nn_scan_dims


############################################################
# Load dataset                                             #
############################################################
ds = xr.open_dataset('Zeffcombo.nc.1')
ds = ds.drop([x for x in ds.coords if x not in ds.dims and x not in ['Zi']])
#ds = xr.open_dataset('4D.nc3')

plot_nn = plot_nn and True
plot_freq = True
plot_ef = True
plot_pf = True

if plot_nn:
    nn = QuaLiKizNDNN.from_json('nn.json')

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
        if nn.feature_min[name] == nn.feature_max[name]:
            # If there are features with a specific value, color the bar red
            # and put marker on value
            start = int(np.argwhere(
                np.isclose(ds[name], nn.feature_min[name], rtol=1e-2)))
            color = 'red'
    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name]).tolist(),
                                       prefix=name + " = ", height=56,
                                       prettify=round, start=start,
                                       color=color)
# Link update event to all sliders
for slider in slider_dict.values():
    slider.on_change('range', updater)

# Display the sliders in two columns
height_block = 300
slidercol1 = widgetbox(list(slider_dict.values())[:len(slider_dict)//2],
                       height=height_block)
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:],
                       height=height_block)
sliderrow = row(slidercol1, slidercol2, sizing_mode='scale_width')

# Create slider to select x-axis
xaxis_name = scan_dims[1]
xaxis_slider = IonRangeSlider(values=scan_dims, height=56, start=scan_dims.index(xaxis_name))
xaxis_slider.on_change('range', swap_x)

toolbar = row(widgetbox([xaxis_slider]), sizing_mode='scale_width')

############################################################
# Create figures                                           #
############################################################
flux_tools = ['box_zoom,pan,zoom_in,zoom_out,reset,save']

x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
figs = {}
plots = []
# Define the flux-like plots (e.g. xaxis_name on the x-axis)
if plot_ef:
    figs['effig']   = Figure(x_axis_label=xaxis_name,
                             y_axis_label='Energy Flux [GB]',
                             height=2*height_block, width=2*height_block,
                             tools=flux_tools, x_range=x_range)
    plots.append(figs['effig'])
if plot_pf:
    figs['pffig']   = Figure(x_axis_label=xaxis_name,
                             y_axis_label='Particle Flux [GB]',
                             height=2*height_block, width=2*height_block,
                             tools=flux_tools, x_range=x_range)
    plots.append(figs['pffig'])

for fig in figs.values():
    hover = HoverTool()
    hover.tooltips = [('x,y', '(@xaxis, @yaxis)')]
    fig.add_tools(hover)
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
    plots.append(freqgrid)


plotrow = row(plots,
              sizing_mode='scale_width', height=2*height_block)

############################################################
# Create legend, style and data sources for fluxplots      #
############################################################
sepcolor = sepcolor[9]
names_particles = ['ele'] + ['Z = ' + str(Zi.data) for Zi in ds['Zi']]

# Define the line styles
color = OrderedDict([('elec', sepcolor[0]),
                     ('nn_elec', sepcolor[0])])
line_dash = OrderedDict([('elec', 'solid'),
                         ('nn_elec', 'dashed')])
legend = OrderedDict([('elec', 'elec'),
                      ('nn_elec', 'nn_elec')])
linenames = ['elec']
if plot_nn:
    linenames.append('nn_elec')

for ii in range(ds.dims['nions']):
    linenames.append('ion' + str(ii))
    color['ion' + str(ii)] = sepcolor[ii + 1]
    line_dash['ion' + str(ii)] = 'solid'
    legend['ion' + str(ii)] = 'Z = ' + str(ds['Zi'].data[ii])
    if plot_nn:
        linenames.append('nn_ion' + str(ii))
        color['nn_ion' + str(ii)] = sepcolor[ii + 1]
        line_dash['nn_ion' + str(ii)] = 'dashed'
        legend['nn_ion' + str(ii)] = 'nn_Z = ' + str(ds['Zi'].data[ii])

sources = {}
# link data sources to figures
for figname in ['effig', 'pffig']:
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
    show(layout)
else:
    curdoc().add_root(layout)
