FROM bokeh/bokeh:0.13.0
MAINTAINER Karel van de Plassche <karelvandeplassche@gmail.com>

RUN apk add --no-cache \
    build-base \
    git

# Install requirements
RUN pip install xarray # This takes a long time! Need more than 512MB RAM!
RUN pip install netcdf4

# Install optional requirements
RUN pip install ipython

# Install NNDB requirements
RUN env NO_SQLITE=1 pip install peewee psycopg2

# Download and install QLKNN-develop source
RUN cd / && git clone https://github.com/karel-van-de-plassche/QLKNN-develop.git
RUN pip install -e /QLKNN-develop

RUN python3 -c "import tornado; print('tornado version=' + tornado.version)"
RUN bokeh info

#sudo docker run --rm -v /home/karel/qlk_data:/qlk_data -v /home/karel/QuaLiKiz-dataslicer:/QuaLiKiz-dataslicer -v /home/karel/QLKNN-develop:/QLKNN-develop -v /home/karel/.pgpass:/root/.pgpass -p 0.0.0.0:5100:5100 -it --name dataslicer dataslicer
