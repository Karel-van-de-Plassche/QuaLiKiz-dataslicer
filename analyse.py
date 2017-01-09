"""
Copyright Dutch Institute for Fundamental Energy Research (2016)
Contributors: Karel van de Plassche (karelvandeplassche@gmail.com)
License: CeCILL v2.1
"""
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import os
from matplotlib.widgets import Slider, Button
import time
from cycler import cycler
from collections import OrderedDict
from math import ceil
from itertools import repeat

import matplotlib.gridspec as gridspec
from IPython import embed
from webalyse import *
from bokeh.plotting import figure, show, reset_output
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox, gridplot
from bokeh.models import HoverTool
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.palettes import Set1 as sepcolor
from bokeh.palettes import Plasma256
import pandas as pd
sepcolor = sepcolor[9]

def takespread(sequence, num, repeat=1):
    length = float(len(sequence))
    for j in range(repeat):
        for i in range(num):
            yield sequence[int(ceil(i * length / num))]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

starttime = time.time()

ds = xr.open_dataset('/home/karel/working/QuaLiKiz/runs/Zeff1.0.compressed.nc')

scan_dims = [name for name in ds.dims if name not in ['nions', 'numsols', 'kthetarhos']]


def plot_subslice(ax, subslice):
        y = subslice
        x, yv = np.meshgrid(subslice['kthetarhos'], subslice[xaxis_name])

        color = takespread(plt.get_cmap('plasma').colors, slice_.dims[xaxis_name])
        ax.set_prop_cycle(cycler('color', color))
        ax.plot(x.T, y.T, marker='o')
    
def extract_plotdata(sel_dict):
    slice_ = ds.sel(method='nearest', **sel_dict)
    plotdata = {}

    
    #plotdata['y_effig'] = np.vstack([np.atleast_2d(slice_['efe_GB'].data), slice_['efi_GB'].data.T]).T
    for prefix in ['ef', 'pf']:
        dict_ = {}
        dict_['xaxis'] = slice_[xaxis_name].data
        for i, efi in enumerate(slice_[prefix + 'i_GB'].T):
            dict_['ion' + str(i)] = efi.data
        dict_['elec'] = slice_[prefix + 'e_GB'].data
        plotdata[prefix + 'fig'] = dict_

    #plotdata['y_pffig'] = np.vstack([np.atleast_2d(slice_['pfe_GB'].data), slice_['pfi_GB'].data.T]).T


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
                #plotdata['x_' + pre + suff].extend(subslice['kthetarhos'].data)
                #plotdata['y_' + pre + suff].extend(subslice.data)
    return plotdata
def plot_data(plotdata):
    #figs['pffig']
    #figs['gamlow']
    #figs['gamhigh']
    #figs['omelow']
    #figs['omehigh']
    return

    #print ('Plotting growthrates/few  ' + str(time.time() - starttime) + ' s')
    #efax.figure.canvas.draw()

def swap_x(attr, old, new):
    #for fig in figs.values():
    #    fig.clear()
    global xaxis_name, sources, figs
    
    try:
        xaxis_name = xaxis_slider.values[new[0]]
        old_xaxis_name = xaxis_slider.values[old[0]]
    except TypeError:
        return
    #slider_dict[old_xaxis_name].disable = False
    #slider_dict[old_xaxis_name].update()
    #slider_dict[xaxis_name].disable = True
    #slider_dict[xaxis_name].update()
    #for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
    for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
        for column_name in sources[figname]:
            sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}
        #for column_name in sources[figname]:
    #        sources[figname][column_name].data = {'xaxis': [], 'yaxis': []}
    for figname in ['effig', 'pffig']:
        for column_name in sources[figname].column_names[1:]:
            sources[name] = []
        #x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
        figs[figname].x_range.start = float(np.min(ds[xaxis_name]))
        figs[figname].x_range.end = float(np.max(ds[xaxis_name]))
        figs[figname].xaxis.axis_label = xaxis_name
    updater(None, None, None)
    
    #for name, slider_list in slider_dict.items():
    #    for i, slider_entry in enumerate(slider_list):
    #        if slider_entry['button'].ax == event.inaxes:
    #            clicked_name = name
    #            clicked_num = i
    #            slider_entry['slider'].poly.set_color('green')
    #            slider_entry['slider'].active = False
    #        else:
    #            slider_entry['slider'].poly.set_color('blue')
    #            slider_entry['slider'].active = True
    #global xaxis_name
    #xaxis_name = clicked_name
    #efax.set_xlabel(xaxis_name)
    #pfax.set_xlabel(xaxis_name)
    #update('')
    #for ax in flux_axes:
    #    ax.relim()      # make sure all the data fits
    #    ax.autoscale()  # auto-scale
    #efax.figure.canvas.draw()

def read_sliders():
    sel_dict = {}
    for name, slider in slider_dict.items():
        if name != xaxis_name:
            sel_dict[name] = slider.values[slider.range[0]]
    return sel_dict

def updater(attr, old, new):
    try:
        sel_dict = read_sliders()
    except TypeError:
        return
    #print(sel_dict)
    plotdata = extract_plotdata(sel_dict)
    #print(plotdata)
    #plot_data(plotdata)
    #print(source.data)
    for figname in ['effig', 'pffig']:
        sources[figname].data = plotdata[figname]
    for figname in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
        for column_name in sources[figname]:
            if column_name in plotdata[figname]:
                sources[figname][column_name].data = plotdata[figname][column_name]
            else:
                sources[figname][column_name].data = {'xaxis': [], 'yaxis': [], 'curval': [], 'cursol': []}
        #sources[name].update()
        #for part, data in zip(range(num_particles), plotdata['y_' + name].T):
            #longname = name + str(part)
            #print(data)
            #sources[longname].data = ColumnDataSource(data)
    #print(sources['effig'].data)

    #figs['omehigh'].title.text = sel_dict['qx']
    #figs['effig'].title.text = 'banana'
    #for name, fig in figs.items():
    #    fig.line([1,2,3],[6,5,4])
    #for slider in slider_dict.values():
    #    print(slider.range)
    #figs['omehigh'].line([1,2,3],[5,4,3])
    #print(sel_dict)
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
#freqgrid = gridplot([[figs['gamlow'], figs['gamhigh']],
#                  [figs['omelow'], figs['omehigh']]])
freqgrid = column(gamrow, omerow, height=2*height_block, sizing_mode='scale_width')

plotrow = row(figs['effig'], figs['pffig'], freqgrid, sizing_mode='scale_width', height=2*height_block)
#plotrow = row(figs['effig'], figs['pffig'], freqgrid, Spacer(height=2*height_block), sizing_mode='scale_width', height=2*height_block)
slidercol1 = widgetbox(list(slider_dict.values())[:len(slider_dict)//2], height=height_block)
slidercol2 = widgetbox(list(slider_dict.values())[len(slider_dict)//2:], height=height_block)
sliderrow = row(slidercol1, slidercol2, sizing_mode='scale_width')

#for name, fig in figs.items():
#    fig.line([1,2,3],[4,5,6])
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
        #dict_['dim' + str(jj) + 'sol' + str(ii)] = []
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
        #print(line)
        #print(column_name)
    figs[figname].add_tools(HoverTool(tooltips=OrderedDict([('val', '@curval'),
                                                            ('sol', '@cursol')]
                                                           )))
#for name in ['effig', 'pffig']:
#    for part in range(num_particles):
#        longname = name + str(part)
#        sources[longname] = ColumnDataSource({xaxis_name:[], name:[]})
#for name in ['gamlow', 'gamhigh', 'omelow', 'omehigh']:
#    sources[name] = ColumnDataSource({xaxis_name:[], 'ylabel':[]})
#    figs[name].scatter(xaxis_name, 'ylabel', source=sources[name])
#    figs[name].multi_line(xaxis_name, 'ylabel', source=sources[name])

for slider in slider_dict.values():
    slider.on_change('range', updater)
xaxis_slider.on_change('range', swap_x)

if __name__ == '__main__':
    embed()
    show(column(plotrow, sliderrow, toolbar, sizing_mode='scale_width'))
else:
    curdoc().add_root(column(plotrow, sliderrow, toolbar, sizing_mode='scale_width'))
