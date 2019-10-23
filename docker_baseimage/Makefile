DOCKERTAG = obspy/seismo-live:latest

all: build push

# build docker base image for seismo-live
build:
	docker build --tag $(DOCKERTAG) - < Dockerfile.seismo-live-base

# login to docker registry (only needs to be done once)
login:
	docker login

# push created image to docker hub
push:
	docker push $(DOCKERTAG)
