## Live Jupyter Notebooks for Seismology

Based on: https://github.com/jupyter/tmpnb

### Installation

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

### Running

To start it, edit the Makefile to set the desired number of Docker workers and available containers and start it with

```bash
make fresh_start
```

### Stop it

```bash
make nuke
```

### Help

```bash
make help
```
