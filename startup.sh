# final steps are supposed to be done as notebook user
# bash scripts in /usr/local/bin/before-notebook.d/ get sourced as notebook
# user

# update notebooks
cd $HOME/seismo_live
git fetch origin
git reset --hard origin/master

# only expose notebooks in the jupyter home dir
cd $HOME
rm -rf $HOME/work
mv $HOME/seismo_live/notebooks/* $HOME/
rm -rf $HOME/seismo_live

# XXX ugly hack to try and work around proj env issues
# XXX https://github.com/conda-forge/basemap-feedstock/issues/30
export PROJ_LIB=/opt/conda/share/proj/
