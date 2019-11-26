# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
# We need to use a unique tag for each update of the base image,
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
# Otherwise if updating the base image with "latest" tag it seems binder just
# caches the old base image and uses that.
# Use the hash tag last used in `make build` for base image, as output after
# running that build
FROM obspy/seismo-live:8d7a0f6a84f7a5
