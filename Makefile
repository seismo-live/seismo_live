.PHONY: build dev nuke super-nuke upload

help:
	@cat Makefile

build:
	-docker pull jupyter/minimal-notebook
	-docker build -t seismolive/all .

fresh_start:
	-export TOKEN=$( head -c 30 /dev/urandom | xxd -p )
	-docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=proxy jupyter/configurable-http-proxy --default-target http://127.0.0.1:9999
	-docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=tmpnb -v /var/run/docker.sock:/docker.sock jupyter/tmpnb python orchestrate.py --container-user=jovyan --command="jupyter notebook --no-browser --port {port} --ip=0.0.0.0 --NotebookApp.base_url={base_path} --NotebookApp.port_retries=0 --NotebookApp.token=\"\" --NotebookApp.disable_check_xsrf=True --ContentsManager.hide_globs=\"['share', '__pycache__', '*.pyc', '*.pyo', '*.so', '*.dylib', '*~']\"" --image='seismolive/all' --pool_size=25 --cull_timeout=1800 --cull_period=300 --redirect-uri='/files/share/overview/index.html' --allow_origin='*'

super-nuke: nuke
	-docker rmi seismolive/all

# Cleanup with fangs
nuke:
	-docker stop `docker ps -aq`
	-docker rm -fv `docker ps -aq`
	-docker images -q --filter "dangling=true" | xargs docker rmi
