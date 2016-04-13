# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
FROM jupyter/scipy-notebook

MAINTAINER Lion Krischer <lion.krischer@gmail.com>

# Install system libs as root.
USER root

RUN apt-get update
RUN apt-get install -y gfortran git

# Rest as jovyan user who is provided by the Jupyter notebook template.
USER jovyan

# Install ObsPy and Instaseis Dependencies.
RUN conda install --yes -c obspy obspy h5py future requests tornado flake8 pytest mock basemap pip jupyter jsonschema
RUN pip install responses
# See https://github.com/ContinuumIO/anaconda-issues/issues/686
# Needed for instaseis.
RUN conda remove libgfortran --force --yes

# Install Instaseis from git.
RUN cd /tmp; git clone https://github.com/krischer/instaseis.git; cd instaseis; pip install -v -e .

# Copy the actual notebooks.
COPY notebooks/ /home/jovyan/work/

# A bit ugly but unfortunately necessary: https://github.com/docker/docker/issues/6119
USER root
RUN chown -R jovyan:users /home/jovyan/work

USER jovyan

# Download the instaseis database.
RUN mkdir -p /home/jovyan/work/Instaseis/data/database
RUN wget -qO- "http://www.geophysik.uni-muenchen.de/~krischer/instaseis/20s_PREM_ANI_FORCES.tar.gz" | tar xvz -C /home/jovyan/work/Instaseis/data/database

# Set a default backend for matplotlib!
RUN mkdir -p ~/.config/matplotlib && touch ~/.config/matplotlib/matplotlibrc && printf "\nbackend: agg\n" >> ~/.config/matplotlib/matplotlibrc
