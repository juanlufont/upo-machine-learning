# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

# Running Jupyter lab in a container

Creating the container
```
docker run \
    -it --name jupyterlab-test \
    -p 8888:8888 \
    -v "$PWD":/home/jovyan/work \
    jupyter/datascience-notebook start.sh jupyter lab
```

Starting an already existing container
```
# -a attach stdout
docker start -a jupyterlab-test
```

If you do not remember access token, look into the logs:
```
docker logs jupyterlab-test
```
