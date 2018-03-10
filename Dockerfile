FROM karelvandeplassche/bokeh
MAINTAINER Karel van de Plassche <karelvandeplassche@gmail.com>

RUN apt-get update \
  && LC_ALL=C DEBIAN_FRONTEND=noninteractive apt-get install -y --force-yes --no-install-recommends \
    gcc \
    g++

RUN pip install xarray # This takes a long time! Need more than 512MB RAM!

RUN LC_ALL=C DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --force-yes --no-install-recommends \
  git \
  ipython3 \
  # Compilation for custon bokeh elements (ionrangeslider)
  nodejs \
  # Headers for python3-netcdf4
  libhdf5-dev libnetcdf-dev pkg-config

RUN pip install --upgrade pip
RUN pip install netcdf4
RUN pip install peewee psycopg2
RUN cd / && git clone https://github.com/karel-van-de-plassche/QLKNN-develop.git
RUN pip install -e /QLKNN-develop

#Run with sudo docker run --rm -v /home/karel/QuaLiKiz-dataslicer:/QuaLiKiz-dataslicer -p 0.0.0.0:5100:5100 -e BOKEH_APP_PATH=/QuaLiKiz-dataslicer/analyse.py -e BOKEH_EXTERNAL_ADDRESS=dataslicer.qualikiz.com -i -t --name dataslicer dataslicer
