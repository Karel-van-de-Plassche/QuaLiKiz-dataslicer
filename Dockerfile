FROM karelvandeplassche/bokeh:0.12.7
MAINTAINER Karel van de Plassche <karelvandeplassche@gmail.com>

RUN apt-get update \
  && LC_ALL=C DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    python3-dev

RUN pip install 'pandas<0.21' --upgrade
RUN pip install netcdf4
RUN pip install xarray # This takes a long time! Need more than 512MB RAM!

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
RUN apt-get update \
  && LC_ALL=C DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    ipython3 \
    # Compilation for custon bokeh elements (ionrangeslider)
    nodejs
    # Headers for python3-netcdf4
    #libhdf5-dev libnetcdf-dev pkg-config

#RUN pip install cftime --upgrade

RUN env NO_SQLITE=1 pip install peewee psycopg2
RUN cd / && git clone https://github.com/karel-van-de-plassche/QLKNN-develop.git
RUN pip install -e /QLKNN-develop

RUN python3 -c "import tornado; print('tornado version=' + tornado.version)"
RUN bokeh info

#Run with sudo docker run --rm -v /home/karel/QuaLiKiz-dataslicer:/QuaLiKiz-dataslicer -p 0.0.0.0:5100:5100 -e BOKEH_APP_PATH=/QuaLiKiz-dataslicer/analyse.py -e BOKEH_EXTERNAL_ADDRESS=dataslicer.qualikiz.com -i -t --name dataslicer dataslicer
