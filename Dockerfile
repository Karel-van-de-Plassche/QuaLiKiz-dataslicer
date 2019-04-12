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

ARG QLKNN_DEVELOP_VERSION=72572f7984522303b2933bc3c1de320e288d9893
ARG QLKNN_FORTRAN_VERSION=e43a5bd006103f014a92e178998798d972629002
ARG QLKNN_HYPER_VERSION=v0.2.0
ARG QUALIKIZ_PYTHONTOOLS_VERSION=9c23cb57414435b7a25079b379430d8e65852651
# Download and install QLKNN-develop source
RUN cd / && git clone https://gitlab.com/karel-van-de-plassche/QLKNN-develop.git &&\
    git -C QLKNN-develop checkout $QLKNN_DEVELOP_VERSION
RUN pip install -e /QLKNN-develop

RUN apk add --no-cache \
    gfortran
RUN pip install f90nml f90wrap
RUN git clone https://gitlab.com/QuaLiKiz-group/QLKNN-fortran.git &&\
    git -C QLKNN-fortran checkout $QLKNN_FORTRAN_VERSION
RUN cp /QLKNN-fortran/src/make.inc/Makefile.docker-debian /QLKNN-fortran/src/Makefile.inc
RUN sed -i '/QLKNNDIR=*/c\QLKNNDIR='/QLKNN-fortran /QLKNN-fortran/src/Makefile.inc
RUN sed -i "s/'string' : 'string',/'string' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
RUN  sed -i "s/'char' : 'string',/'char' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
RUN sed -i "s/'signed_char' : 'string',/'signed_char' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
RUN cd /QLKNN-fortran/src && make FC_WRAPPER='$(FC)' python
RUN mv /QLKNN-fortran/src/_QLKNNFORT.cpython-36m-x86_64-linux-gnu.so /QLKNN-fortran/src/QLKNNFORT.py /QLKNN-develop/qlknn/models

RUN cd /QLKNN-fortran && git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git &&\
    git -C qlknn-hyper checkout $QLKNN_HYPER_VERSION
RUN mkdir /QLKNN-fortran/lib
RUN python QLKNN-fortran/tools/json_nn_to_namelist.py /QLKNN-fortran/qlknn-hyper/ /QLKNN-fortran/lib

RUN git clone https://gitlab.com/qualikiz-group/qualikiz-pythontools.git &&\
    git -C qualikiz-pythontools checkout $QUALIKIZ_PYTHONTOOLS_VERSION
RUN pip install -e qualikiz-pythontools

RUN python3 -c "import tornado; print('tornado version=' + tornado.version)"
RUN bokeh info

#sudo docker run --rm -v /home/karel/qlk_data:/qlk_data -v /home/karel/QuaLiKiz-dataslicer:/QuaLiKiz-dataslicer -v /home/karel/QLKNN-develop:/QLKNN-develop -v /home/karel/.pgpass:/root/.pgpass -p 0.0.0.0:5100:5100 -it --name dataslicer dataslicer
