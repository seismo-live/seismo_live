# Fork from a jupyter provided template. Its a scientific stack with a conda
# environment. Defaults to Python 3 but also has Python 2. For now we'll only
# install libs on Python 3.
# according to docs, we need to explicitly specify the tag, see
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
# however, for me using the hash actually did NOT work and using "latest" worked
FROM obspy/seismo-live:latest
#FROM obspy/seismo-live:9bcbfc7afe6eaf29a24e878fe2d658bb46797eb522c190bc0f06f2339546ead5
#FROM obspy/seismo-live:9bcbfc7afe6e

# from https://success.docker.com/article/use-a-script-to-initialize-stateful-container-data
COPY docker-entrypoint.sh /usr/local/bin/
RUN ln -s /usr/local/bin/docker-entrypoint.sh / # backwards compat
ENTRYPOINT ["docker-entrypoint.sh"]
