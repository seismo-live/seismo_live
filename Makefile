.PHONY: build dev nuke super-nuke upload

help:
	@cat Makefile

build:
	-docker pull jupyter/minimal
	-docker build -t seismolive/all .

fresh_start:
	-export TOKEN=$( head -c 30 /dev/urandom | xxd -p )
	-docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=proxy jupyter/configurable-http-proxy --default-target http://127.0.0.1:9999
	-docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN -v /var/run/docker.sock:/docker.sock jupyter/tmpnb python orchestrate.py --image='seismolive/all' --command="ipython notebook --NotebookApp.base_url={base_path} --ip=0.0.0.0 --port {port}" --allow_origin='*' --max_dock_workers=4 --pool_size=50 --cull_timeout=1800 --cull_period=300

super-nuke: nuke
	-docker rmi seismolive/all

# Cleanup with fangs
nuke:
	-docker stop `docker ps -aq`
	-docker rm -fv `docker ps -aq`
	-docker images -q --filter "dangling=true" | xargs docker rmi
