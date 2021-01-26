#!make

include jupyterlab.env
export

.PHONY: run run_daemon stop start logs rm check_rm purge login

run:
	mkdir -p $(shell pwd)/.jupyter
	mkdir -p $(shell pwd)/work
	docker run \
		-it --name $(DOCKER_CONTAINER_NAME) \
		-p $(CONT_PORT):$(HOST_PORT) \
		-v $(shell pwd)/notebook:/home/jovyan/work \
		-v $(shell pwd)/.jupyter:/home/jovyan/.jupyter \
		$(DOCKER_IMAGE) start.sh jupyter lab

stop:
	docker stop $(DOCKER_CONTAINER_NAME)

start:
	docker start -a $(DOCKER_CONTAINER_NAME)

logs:
	docker logs $(DOCKER_CONTAINER_NAME)

rm: check_rm
	docker stop $(DOCKER_CONTAINER_NAME)
	docker rm $(DOCKER_CONTAINER_NAME)

check_rm:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

purge: rm
	rm -rf $(shell pwd)/.jupyter

login:
	docker exec -it $(DOCKER_CONTAINER_NAME) /bin/bash
