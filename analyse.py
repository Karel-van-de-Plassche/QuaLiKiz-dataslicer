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
from cycler import cycler
from collections import OrderedDict
from math import ceil
from itertools import repeat

from IPython import embed
from ionrangeslider.ionrangeslider import IonRangeSlider
from bokeh.plotting import figure, show, reset_output, Figure
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox
from bokeh.models import HoverTool
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.palettes import Set1 as sepcolor
from bokeh.palettes import Plasma256
sepcolor = sepcolor[9]


def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]


def extract_plotdata(sel_dict):
    slice_ = ds.sel(method='nearest', **sel_dict)
    plotdata = {}

    for prefix in ['ef', 'pf']:
        dict_ = {}
        dict_['xaxis'] = slice_[xaxis_name].data
        for i, efi in enumerate(slice_[prefix + 'i_GB'].T):
            dict_['ion' + str(i)] = efi.data
        dict_['elec'] = slice_[prefix + 'e_GB'].data
        plotdata[prefix + 'fig'] = dict_

    for suff in ['low', 'high']:
        for pre in ['gam', 'ome']:
            plotdata[pre + suff] = {}
    for numsol in slice_['numsols'].data:
        for suff in ['low', 'high']:
            for pre in ['gam', 'ome']:
                if suff == 'low':
                    kthetarhos = slice(None, kthetarhos_cutoff)
                else:
                    kthetarhos = slice(kthetarhos_cutoff, None)
                subslice = slice_[pre + '_GB'].sel(numsols=numsol, kthetarhos=kthetarhos)
                subslice = subslice.where(subslice != 0)

                for ii, subsubslice in enumerate(subslice):
                    plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))] = {}
                    plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['xaxis'] = subslice['kthetarhos'].data
                    plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['yaxis'] = subsubslice.data
                    plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['curval'] = np.full_like(subslice['kthetarhos'], subsubslice[xaxis_name].data)
                    plotdata[pre + suff]['dim' + str(ii) + 'sol' + str(int(numsol))]['cursol'] = np.full_like(subslice['kthetarhos'], int(numsol))
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
        for column_name in sources[figname].column_names[1:]:
            sources[name] = []

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


def updater(attr, old, new):
    try:
        # This will (silently) fail when the sliders are not initialized yet
        sel_dict = read_sliders()
    except TypeError:
        return

    plotdata = extract_plotdata(sel_dict)
    for figname in ['effig', 'pffig']:
        sources[figname].data = plotdata[figname]
    for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
        for column_name in sources[figname]:
            if column_name in plotdata[figname]:
                sources[figname][column_name].data = plotdata[figname][column_name]
            else:
                sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}


ds = xr.open_dataset('/mnt/hdd/Zeff_combined.nc')

scan_dims = [name for name in ds.dims if name not in ['nions', 'numsols', 'kthetarhos']]
# Create slider dict
round = CustomJS(code="""
         var f = cb_obj
         f = Number(f.toPrecision(2))
         return f
     """)
slider_dict = OrderedDict()
numslider = 0
xaxis_name = 'Ate'
kthetarhos_cutoff = 1
for name in scan_dims:
    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name]).tolist(), prefix=name + " = ", height=56, prettify=round)
#slider_dict[xaxis_name].disable = True

xaxis_slider = IonRangeSlider(values=scan_dims, height=56)
toolbar = row(widgetbox([xaxis_slider]), sizing_mode='scale_width')


height_block = 300
flux_tools = 'box_zoom,pan,zoom_in,zoom_out,reset,save,hover'
freq_tools = 'save'

x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
figs = {}
figs['effig']   = Figure(x_axis_label=xaxis_name,   y_axis_label='Energy Flux [GB]',
                        height=2*height_block, width=2*height_block,
                        tools=flux_tools, x_range=x_range)
figs['pffig']   = Figure(x_axis_label=xaxis_name,   y_axis_label='Particle Flux [GB]',
                        height=2*height_block, width=2*height_block,
                        tools=flux_tools, x_range=x_range)
figs['gamlow']  = Figure(x_axis_label=' ',          y_axis_label='Growth Rates [GB]',
                        height=height_block,   width=height_block,
                        tools=freq_tools, x_range=[0, kthetarhos_cutoff])
figs['gamhigh'] = Figure(x_axis_label=' ',          y_axis_label=' ',
                        height=height_block,   width=height_block,
                        tools=freq_tools, x_range=[kthetarhos_cutoff, float(ds['kthetarhos'].max())])
figs['omelow']  = Figure(x_axis_label='kthetarhos', y_axis_label='Frequencies [GB]',
                        height=height_block,   width=height_block,
                        tools=freq_tools, x_range=[0, kthetarhos_cutoff])
figs['omehigh'] = Figure(x_axis_label='kthetarhos', y_axis_label=' ',
                         height=height_block,   width=height_block,
                         tools=freq_tools, x_range=[kthetarhos_cutoff, float(ds['kthetarhos'].max())])
gamrow = row(figs['gamlow'], figs['gamhigh'], height=height_block, width=height_block, sizing_mode='scale_width')
omerow = row(figs['omelow'], figs['omehigh'], height=height_block, width=height_block, sizing_mode='scale_width')

freqgrid = column(gamrow, omerow, height=2*height_block, sizing_mode='scale_width')

plotrow = row(figs['effig'], figs['pffig'], freqgrid, sizing_mode='scale_width', height=2*height_block)
slidercol1 = widgetbox(list(slider_dict.values())[:len(slider_dict)//2], height=height_block)
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:], height=height_block)
sliderrow = row(slidercol1, slidercol2, sizing_mode='scale_width')

sources = {}

names_particles = ['ele'] + ['Z = ' + str(Zi.data) for Zi in ds['Zi']]
dict_ = OrderedDict([('xaxis', []),
                     ('elec', [])])
for ii in range(ds.dims['nions']):
    dict_['ion' + str(ii)] = []

for figname in ['effig', 'pffig']:
    sources[figname] = ColumnDataSource(dict_)
    for ii, column_name in enumerate(sources[figname].column_names[1:]):
        figs[figname].scatter('xaxis', column_name, source=sources[figname], color=sepcolor[ii], legend=names_particles[ii])
        figs[figname].line('xaxis', column_name, source=sources[figname], color=sepcolor[ii], legend=names_particles[ii])
        figs[figname].legend.location = 'top_left'

max_num = 0
for scan_dim in scan_dims:
    num = ds.dims[scan_dim]
    if num > max_num:
        max_num = num
dict_ = OrderedDict([('xaxis', [])])
linenames = []
for ii in range(max_num):
    for jj in range(ds.dims['numsols']):
        linenames.append('dim' + str(ii) + 'sol' + str(jj))

renderers = []
for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
    seqcolor = takespread(Plasma256, max_num, repeat=int(ds.dims['numsols']))
    sources[figname] = OrderedDict()
    for color, column_name in zip(seqcolor, linenames):
        source = sources[figname][column_name] = ColumnDataSource({'xaxis': [],
                                                                   'yaxis': [],
                                                                   'curval': [],
                                                                   'cursol': []})
        opts = dict(source=source, color=color, alpha=.5,name=column_name, hover_color=color)
        sc = figs[figname].scatter('xaxis', 'yaxis', **opts)
        line = figs[figname].line('xaxis', 'yaxis', **opts)
    figs[figname].add_tools(HoverTool(tooltips=OrderedDict([('val', '@curval'),
                                                            ('sol', '@cursol')]
                                                          )))

for slider in slider_dict.values():
    slider.on_change('range', updater)
xaxis_slider.on_change('range', swap_x)

if __name__ == '__main__':
    embed()
    show(column(plotrow, sliderrow, toolbar, sizing_mode='scale_width'))
else:
    curdoc().add_root(column(plotrow, sliderrow, toolbar, sizing_mode='scale_width'))
