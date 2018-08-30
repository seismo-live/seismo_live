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

# Install ObsPy and Instaseis.
# XXX temporarily pin matplotlib due to obspy/obspy#2097 not yet in a release
# XXX version
RUN conda install --yes -c conda-forge obspy future requests tornado flake8 pytest mock basemap pip jupyter jsonschema basemap-data-hires instaseis pandas seaborn 'matplotlib<2.2'

# Install the rate and state toolkit.
RUN pip install https://github.com/jrleeman/rsfmodel/archive/master.zip

# Install jupyter lab.
RUN pip install jupyterlab
RUN jupyter serverextension enable --py jupyterlab

# Install the jupyter dashboards.
RUN pip install jupyter_dashboards
RUN jupyter dashboards quick-setup --sys-prefix
RUN jupyter nbextension enable jupyter_dashboards --py --sys-prefix

# Install the code folding plugin.
RUN conda install --yes -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main

# Copy the actual notebooks.
COPY notebooks/ /home/jovyan

# A bit ugly but unfortunately necessary: https://github.com/docker/docker/issues/6119
USER root
RUN chown -R jovyan:users /home/jovyan

USER jovyan

# This might exist locally and will thus be copied to the docker image...
RUN rm -rf /home/jovyan/Instaseis-Syngine/data/database
# Download the instaseis database.
RUN mkdir -p /home/jovyan/Instaseis-Syngine/data/database
RUN wget -qO- "http://www.geophysik.uni-muenchen.de/~krischer/instaseis/20s_PREM_ANI_FORCES.tar.gz" | tar xvz -C /home/jovyan/Instaseis-Syngine/data/database

# Set a default backend for matplotlib!
RUN mkdir -p ~/.config/matplotlib && touch ~/.config/matplotlib/matplotlibrc && printf "\nbackend: nbagg\n" >> ~/.config/matplotlib/matplotlibrc

# Build the font cache so its already done in the notebooks.
RUN python -c "from matplotlib.font_manager import FontManager; FontManager()"

# Ignore all Python warnings because they look ugly in the docs.
ENV PYTHONWARNINGS ignore
