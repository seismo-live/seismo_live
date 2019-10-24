DOCKERREPO = obspy/seismo-live

all: build push

# build docker base image for seismo-live
build:
	docker build --tag $(DOCKERREPO):`git rev-parse --short=14 HEAD` - < Dockerfile.seismo-live-base

# login to docker registry (only needs to be done once)
login:
	docker login

# push created image to docker hub
push:
	docker push $(DOCKERREPO)
