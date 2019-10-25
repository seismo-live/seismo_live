DOCKERREPO = obspy/seismo-live
CLEANWARN = This will delete stopped docker containers, untagged images and all local images in target docker repository

all: build push

# remove stopped containers, untagged images and all local content in our
# target docker repo, to ensure a clean run
clean: check_clean
	docker container prune -f
	docker image prune -f --filter "dangling=true"
	if [ `docker images $(DOCKERREPO) -q | wc -l` -gt 0 ]; then docker rmi `docker images $(DOCKERREPO) -q`; fi

check_clean:
	@echo -n "$(CLEANWARN) \"$(DOCKERREPO)\". Continue? [y/N] " && read ans && [ $${ans:-N} = y ]

.PHONY: clean check_clean

# build docker base image for seismo-live
build: clean
	docker build --tag $(DOCKERREPO):`git rev-parse --short=14 HEAD` - < Dockerfile.seismo-live-base 2>&1 | tee build_`git rev-parse --short=14 HEAD`.log

# login to docker registry (only needs to be done once)
login:
	docker login

# push created image to docker hub
push:
	docker push $(DOCKERREPO)
