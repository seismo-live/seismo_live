# -*- coding: utf-8 -*-
#!/bin/bash

# final steps to be done as notebook user
sudo -u jovyan -- -sh -c <<EOT

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

# XXX just for testing
touch $HOME/testfile
EOT

# XXX ugly hack to try and work around proj env issues
# XXX https://github.com/conda-forge/basemap-feedstock/issues/30
export PROJ_LIB=/opt/conda/share/proj/
