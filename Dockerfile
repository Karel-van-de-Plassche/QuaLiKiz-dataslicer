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

ARG QLKNN_DEVELOP_VERSION=b5877b1eebdf0d226eb195b8cd706b0b660fc15e
ARG QLKNN_FORTRAN_VERSION=371f575ea806cd89341ed62e458bae0ed8e528b3
ARG QLKNN_HYPER_VERSION=v0.5.0
ARG QUALIKIZ_PYTHONTOOLS_VERSION=b42f7aa1e475807c25683cd18f984553aff8fbbe
# Download and install QLKNN-develop source
RUN cd / && git clone https://gitlab.com/karel-van-de-plassche/QLKNN-develop.git &&\
    git -C QLKNN-develop checkout $QLKNN_DEVELOP_VERSION
RUN pip install -e /QLKNN-develop

RUN apk add --no-cache \
    gfortran
RUN pip install f90nml f90wrap
RUN git clone https://gitlab.com/QuaLiKiz-group/QLKNN-fortran.git &&\
    git -C QLKNN-fortran checkout $QLKNN_FORTRAN_VERSION
#RUN cp /QLKNN-fortran/src/make.inc/Makefile.docker-debian /QLKNN-fortran/src/Makefile.inc
#RUN sed -i '/QLKNNDIR=*/c\QLKNNDIR='/QLKNN-fortran /QLKNN-fortran/src/Makefile.inc
#RUN sed -i "s/'string' : 'string',/'string' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
#RUN  sed -i "s/'char' : 'string',/'char' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
#RUN sed -i "s/'signed_char' : 'string',/'signed_char' : 'str',/" /opt/conda/lib/python3.6/site-packages/f90wrap/fortran.py
RUN git -C /QLKNN-fortran submodule update --init
RUN cd /QLKNN-fortran && make TOOLCHAIN=gcc TUBSCFG_MPI=0 python
RUN mv /QLKNN-fortran/build/gcc-release-default/_QLKNNFORT.cpython-36m-x86_64-linux-gnu.so /QLKNN-fortran/build/gcc-release-default/QLKNNFORT.py /QLKNN-develop/qlknn/models

RUN cd /QLKNN-fortran && git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git &&\
    git -C qlknn-hyper checkout $QLKNN_HYPER_VERSION
RUN mkdir -p /QLKNN-fortran/lib/src/qlknn-hyper-namelists
RUN python QLKNN-fortran/tools/json_nn_to_namelist.py /QLKNN-fortran/qlknn-hyper/ /QLKNN-fortran/lib/src/qlknn-hyper-namelists

RUN git clone https://gitlab.com/qualikiz-group/qualikiz-pythontools.git &&\
    git -C qualikiz-pythontools checkout $QUALIKIZ_PYTHONTOOLS_VERSION
RUN pip install -e qualikiz-pythontools

RUN python3 -c "import tornado; print('tornado version=' + tornado.version)"
RUN bokeh info

#sudo docker run --rm -v /home/karel/qlk_data:/qlk_data -v /home/karel/QuaLiKiz-dataslicer:/QuaLiKiz-dataslicer -v /home/karel/QLKNN-develop:/QLKNN-develop -v /home/karel/.pgpass:/root/.pgpass -p 0.0.0.0:5100:5100 -it --name dataslicer dataslicer
