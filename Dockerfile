# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
FROM jupyter/scipy-notebook

MAINTAINER Lion Krischer <lion.krischer@gmail.com>

# Install system libs as root.
USER root

RUN apt-get install -y gfortran git


# Rest as jovyan user who is provided by the Jupyter notebook template.
USER jovyan

# Install ObsPy and Instaseis Dependencies.
RUN conda install --yes -c obspy obspy netcdf4 future requests tornado flake8 pytest mock basemap pip jupyter
RUN pip install responses

RUN git clone https://github.com/krischer/instaseis.git
RUN cd instaseis; pip install -v -e .

COPY notebooks/ /home/jovyan/work/
