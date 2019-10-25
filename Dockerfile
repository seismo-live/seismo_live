# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
# We need to use a unique tag for each update of the base image,
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
# Otherwise if updating the base image with "latest" tag it seems binder just
# caches the old base image and uses that.
# Use the hash tag last used in `make build` for base image, as output after
# running that build
FROM obspy/seismo-live:aa9c9f57d71272

USER jovyan
# update notebooks to current master
RUN cd $HOME/seismo_live && git fetch origin && git reset --hard origin/master
# only expose notebooks in the jupyter home dir, delete everything else
RUN cd $HOME && rm -rf $HOME/work && mv $HOME/seismo_live/notebooks/* $HOME/ && rm -rf $HOME/seismo_live

# XXX ugly hack to try and work around proj env issues
# XXX https://github.com/conda-forge/basemap-feedstock/issues/30
ENV PROJ_LIB=/opt/conda/share/proj/
