# -*- coding: utf-8 -*-
#!/bin/bash

# update notebooks
cd $HOME/seismo_live
#git fetch origin
#git reset --hard origin/master
git pull

# only expose notebooks in the jupyter home dir
ls -l *
cd $HOME
rm -rf $HOME/work
mv $HOME/seismo_live/notebooks/* $HOME/
rm -rf $HOME/seismo_live
ls -l *
ls -l $HOME

# XXX ugly hack to try and work around proj env issues
# XXX https://github.com/conda-forge/basemap-feedstock/issues/30
export PROJ_LIB=/opt/conda/share/proj/

# Continue with normal startup of jupyter
# see https://github.com/jupyter/docker-stacks/blob/1386e20468332f32a028c6224bbd8439eb406ee4/base-notebook/Dockerfile#L120
exec /usr/local/bin/start-notebook.sh "$@"
