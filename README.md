## Live Jupyter Notebooks for Seismology

### Building the Website

It is currently a two step procedure:

First build all notebooks. This takes the jupytext files, creates an exercise
and solution version of each if necessary, converts both to ipynb, runs them,
and renders to the outputs to HTML.

All the output will be store in in `built_notebooks`.

```bash
$ python conf/convert_to_ipynb.py notebooks built_notebooks
```

The second step takes these outputs and create the final website which is
stored in `final_website`:

```bash
$ python conf/build_website.py built_notebooks final_website
```

### Contributing

We intend seismo-live to be a place to collect all kinds of tutorial and notebooks related to seismology so contributions are gladly accepted and actually crucial for the success of the whole project. To contribute make sure you have the same installation, especially **Python version 3.5**, as documented below. If you require additional packages please mention it in your pull request. Once your environment is setup, create your new notebooks and send us a pull request. Tutorials on how to do that can be found [here](https://yangsu.github.io/pull-request-tutorial/) and [here](https://www.thinkful.com/learn/github-pull-request-tutorial/) and lots of other places online. If you need help, don't hesitate to contact us.

**New contributors, please sign this:** https://www.clahub.com/agreements/krischer/seismo_live

### Server Installation

This explains how to install seismo-live on a server. For a local installation see below.

Based on: https://github.com/jupyter/tmpnb

#### Installation

Install Docker for you platform: http://docs.docker.com/installation

Don't use the repository version as that might be very old.

```bash
# Add the current user to the docker group
sudo usermod -a -G docker USERNAME

# git is also required, install if not available.
sudo apt-get install git

# Furthermore `make` must be available.
sudo apt-get install build-essential

# Checkout the repository (a shallow clone is enough)
git clone --depth=1 https://github.com/krischer/seismo_live.git

cd seismo_live
# Can take quite a while!
make build
```

#### Running

To start it, edit the Makefile to set the desired number of Docker workers and available containers and start it with

```bash
make fresh_start
```

#### Stop it

```bash
make nuke
```

#### Help

```bash
make help
```

### Local Installation

You might be interested in running the notebooks locally on your own computer. A big advantage is that any changes you make will no longer be deleted. You can also contribute changes you made (or entirely new notebooks) back to the seismo-live project!

The notebooks as of now require:

- Python 3.5
- The scientific Python stack (NumPy, SciPy, matplotlib)
- The Jupyter notebooks
- ObsPy >= 1.0.1
- Instaseis

We recommend to install ObsPy via Anaconda as written [in its installation instructions](https://github.com/obspy/obspy/wiki/Installation-via-Anaconda). Then install Instaseis (does not work on Windows) according to [its documentation](http://instaseis.net/#installation). Finally install the Jupyter project with

```bash
$ conda install jupyter
```

Now just clone the project from Github, cd to the correct folder and launch the notebook server.

```bash
$ git clone --depth=1 https://github.com/krischer/seismo_live.git
$ cd seismo_live/notebooks
$ jupyter-notebook
```

Please note that the Instaseis notebooks require a local database symlinked to `seismo_live/notebooks/Instaseis/data/database`. You could get one for example with:

```bash
$ wget -qO- "http://www.geophysik.uni-muenchen.de/~krischer/instaseis/20s_PREM_ANI_FORCES.tar.gz" | tar xvz -C 20s_PREM_ANI_INSTASEIS_DB
```
