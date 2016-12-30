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

import matplotlib.gridspec as gridspec
from IPython import embed
from webalyse import *
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, layout, gridplot, Spacer, widgetbox, gridplot
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource

def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

starttime = time.time()

ds = xr.open_dataset('/home/karel/Zeff_combined.nc')

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

    
    plotdata['x_flux'] = slice_[xaxis_name].data
    plotdata['y_ef'] = np.vstack([np.atleast_2d(slice_['efe_GB'].data), slice_['efi_GB'].data.T]).T
    plotdata['y_ef'] = slice_['efe_GB'].data
    plotdata['y_pf'] = np.vstack([np.atleast_2d(slice_['pfe_GB'].data), slice_['pfi_GB'].data.T]).T


    for suff in ['low', 'high']:
        for pre in ['gam', 'ome']:
            for ax in ['x', 'y']:
                plotdata[ax + '_' + pre + suff] = []
    for numsol in slice_['numsols']:
        for suff in ['low', 'high']:
            for pre in ['gam', 'ome']:
                if suff == 'low':
                    kthetarhos = slice(None, kthetarhos_cutoff)
                else:
                    kthetarhos = slice(kthetarhos_cutoff, None)
                subslice = slice_[pre + '_GB'].sel(numsols=numsol, kthetarhos=kthetarhos)
                subslice = subslice.where(subslice != 0)
                plotdata['x_' + pre + suff].extend(subslice['kthetarhos'].data)
                plotdata['y_' + pre + suff].extend(subslice.data)
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

def swap_x(event):
    for name, slider_list in slider_dict.items():
        for i, slider_entry in enumerate(slider_list):
            if slider_entry['button'].ax == event.inaxes:
                clicked_name = name
                clicked_num = i
                slider_entry['slider'].poly.set_color('green')
                slider_entry['slider'].active = False
            else:
                slider_entry['slider'].poly.set_color('blue')
                slider_entry['slider'].active = True
    global xaxis_name
    xaxis_name = clicked_name
    efax.set_xlabel(xaxis_name)
    pfax.set_xlabel(xaxis_name)
    update('')
    for ax in flux_axes:
        ax.relim()      # make sure all the data fits
        ax.autoscale()  # auto-scale
    efax.figure.canvas.draw()

def read_sliders():
    sel_dict = {}
    for name, slider in slider_dict.items():
        if name != xaxis_name:
            sel_dict[name] = slider.values[slider.range[0]]
    return sel_dict

def updater(attr, old, new):
    sel_dict = read_sliders()
    #print(sel_dict)
    plotdata = extract_plotdata(sel_dict)
    #print(plotdata)
    #plot_data(plotdata)
    #print(source.data)
    sources['effig'].data = {xaxis_name: plotdata['x_flux'],
                             'ylabel': plotdata['y_ef']}
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
    slider_dict[name] = IonRangeSlider(values=np.unique(ds[name]).tolist(), prefix=name + " = ", height=60, prettify=round, force_edges=True)


figs = {}
height_block =200 
x_range = [float(np.min(ds[xaxis_name])), float(np.max(ds[xaxis_name]))]
figs['effig'] = figure(title="Heat Flux", x_axis_label=xaxis_name, y_axis_label='ylab', height=2*height_block, width=2*height_block, x_range=x_range)
figs['pffig'] = figure(title="Particle Flux", x_axis_label=xaxis_name, y_axis_label='ysab', height=2*height_block, width=2*height_block, x_range=x_range)
figs['gamlow'] = figure(y_axis_label='gam_GB' , height=height_block, width=height_block)
figs['gamhigh'] = figure(y_axis_label=' ', height=height_block, width=height_block)
figs['omelow'] = figure(x_axis_label='kthetarhos', y_axis_label='ome_GB' , height=height_block, width=height_block)
figs['omehigh'] = figure(x_axis_label='kthetarhos',y_axis_label=' ',  height=height_block, width=height_block)
gamrow = row(figs['gamlow'], figs['gamhigh'], height=height_block, sizing_mode='scale_width')
omerow = row(figs['omelow'], figs['omehigh'], height=height_block, sizing_mode='scale_width')
#freqgrid = gridplot([[figs['gamlow'], figs['gamhigh']],
#                  [figs['omelow'], figs['omehigh']]])
freqgrid = column(gamrow, omerow, height=2*height_block, sizing_mode='scale_width')

plotrow = row(figs['effig'], figs['pffig'], freqgrid, Spacer(height=2*height_block), sizing_mode='scale_width', height=2*height_block)
sliderrow = widgetbox(children=slider_dict.values(), height=height_block)
#for name, fig in figs.items():
#    fig.line([1,2,3],[4,5,6])

sources = {}
for name in ['effig', 'pffig', 'gamlow', 'gamhigh', 'omelow', 'omehigh']:
    sources[name] = ColumnDataSource({xaxis_name:[], 'ylabel':[]})
    figs[name].scatter(xaxis_name, 'ylabel', source=sources[name])
    figs[name].line(xaxis_name, 'ylabel', source=sources[name])
for slider in slider_dict.values():
    slider.on_change('range', updater)
if __name__ == '__main__':
    embed()
    show(column(plotrow, sliderrow, sizing_mode='scale_width'))
else:
    curdoc().add_root(column(plotrow, sliderrow, sizing_mode='scale_width'))
