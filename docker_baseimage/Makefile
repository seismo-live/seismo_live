DOCKERREPO = obspy/seismo-live
CLEANWARN = This will delete stopped docker containers, untagged images and all local images in target docker repository
COMMITHASH = $(shell git rev-parse --short=14 HEAD)
IMAGE = $(DOCKERREPO):$(COMMITHASH)

all: build push run
	@echo done working on image $(IMAGE)

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
	@echo building image $(IMAGE)
	docker build --tag $(IMAGE) - < Dockerfile.seismo-live-base 2>&1 | tee build_$(COMMITHASH).log

# login to docker registry (only needs to be done once)
login:
	docker login

# push created image to docker hub
push:
	@echo pushing image $(IMAGE)
	docker push $(IMAGE)

# run Jupyter in docker image locally
run:
	@echo running image $(IMAGE)
	@# only run the image if its available locally
	if [ `docker images -q $(IMAGE) | wc -l` -eq 0 ]; then echo "image $(IMAGE) not found locally"; exit 1; fi
	docker run --rm -p 8888:8888 $(IMAGE)
