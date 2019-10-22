# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
# need to explicitly specify the tag, see
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
FROM obspy/seismo-live:latest
#FROM obspy/seismo-live:9bcbfc7afe6eaf29a24e878fe2d658bb46797eb522c190bc0f06f2339546ead5
#FROM obspy/seismo-live:9bcbfc7afe6e

MAINTAINER Tobias Megies <megies@geophysik.uni-muenchen.de>

# update notebooks
RUN cd /home/jovyan/seismo_live && git fetch origin && git reset --hard origin/master

# XXX ugly hack to try and work around proj env issues
# XXX https://github.com/conda-forge/basemap-feedstock/issues/30
ENV PROJ_LIB /opt/conda/share/proj/
