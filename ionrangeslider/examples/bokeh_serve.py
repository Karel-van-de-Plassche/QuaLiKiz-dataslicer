from bokeh.layouts import column
from bokeh.models import Slider, CustomJS, ColumnDataSource
from bokeh.plotting import Figure
from bokeh.io import curdoc

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ionrangeslider.ionrangeslider import IonRangeSlider

x = [x*0.005 for x in range(2, 198)]
y = x

source = ColumnDataSource(data=dict(x=x, y=y))

plot = Figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6, color='#ed5565')

callback_single = CustomJS(args=dict(source=source), code="""
        var data = source.data;
        var f = cb_obj.value
        x = data['x']
        y = data['y']
        for (i = 0; i < x.length; i++) {
            y[i] = Math.pow(x[i], f)
        }
        source.trigger('change');
    """)

callback_ion = CustomJS(args=dict(source=source), code="""
        var data = source.data;
        var f = cb_obj.range
        x = data['x']
        y = data['y']
        pow = (Math.log(y[100])/Math.log(x[100]))
        console.log(pow)
        delta = (f[1]-f[0])/x.length
        for (i = 0; i < x.length; i++) {
            x[i] = delta*i + f[0]
            y[i] = Math.pow(x[i], pow)
        }
        source.trigger('change');
    """)

prettifier = CustomJS(code="""
        var f = cb_obj
        f = Number(f.toPrecision(2))
        return f
    """)
# Native bokeh slider
slider = Slider(start=0, end=5, step=0.1, value=1, title="Bokeh Slider - Power", callback=callback_single)
# IonRangeSlider from http://bokeh.pydata.org/en/latest/docs/user_guide/extensions_gallery/widget.html
ion_range_slider = IonRangeSlider(start=0.01, end=0.99, step=0.01,
                                  range=(min(x), max(x)),
                                  title='Ion Range Slider - Range',
                                  callback=callback_ion,
                                  callback_policy='continuous',
                                  slider_type='double')
# IonRangeSlider with extended capabilities
ion_range_slider2 = IonRangeSlider(values=[0, 1e-4, 0.00999993],
                                   title='Extended Slider',
                                   prettify_enabled=True, prettify=prettifier,
                                   grid=False, color='green',
                                   force_edges=True)
layout = column(plot, slider, ion_range_slider, ion_range_slider2)
curdoc().add_root(layout)
