# README #

Este repositorio contiene código R, documentación y herramientas auxiliares para llevar a cabo la práctica de minería de datos y machine learning para la asignatura *Análisis de datos e investigación en historia política y social* de máster de *Historia y Humanidades Digitales*.

La práctica se ha realizado usando un [Jupyter notebook](https://jupyter.org) con un kernel para lenguaje R y comentarios embebidos en formato Markdown.

El repositorio incluye scripts para iniciar y configurar un contenedor Docker con todo el software necesario para ejecutar el Jupyter notebook, facilitando la reproducibilidad de los cálculos.

## Contenido del repositorio
* `Makefile`, define reglas para el manejo de un contenedor Docker para ejecutar Jupyter notebooks con soporte para lenguaje R.
* `jupiterlab.env`, contains the configuration for the Docker container.
* `notebook/data_mining.ipynb`, notebook con el código R y documentación relacionada con la resolución de la práctica propuesta.

## Ejecutando el notebook en una instalación local de Jupyter
El notebook puede ser directamente abierto por cualquier instalación local de Jupyter (mediante sistema de paquetes del sistema operativo o herramientas como [Conda](https://docs.conda.io/en/latest/)) y que ademas incluya soporte para lenguaje R y las siguientes dependencias de R:
* `fselector`
* `rweka`
* `e1071`
* `C50`
* `microbenchmark`


## Ejecutando el notebook en un contenedor Docker
Es posible ejecutar el notebook con el código de la práctica en un contenedor Docker y gestionado a través del archivo `Makefile` incluido en el repositorio. Este método asegura que todas las dependencias estarán presentes y contenidas en un contenedor fácilmente gestionable con sencillos comandos basados en `make`.

**NOTA:** el contenedor creado a través del archivo `Makefile` es persistente, lo que significa que a menos que sea explícitamente borrado, mantiene todas las configuraciones y paquetes instalados entre sucesivas ejecuciones.

### Dependencias software
Para poder ejecutar el notebook en un contenedor Docker es necesario:
* una instalación funcional de [Docker](https://docs.docker.com/get-docker/)
* herramienta `make` para facilitar la gestión del contenedor (i.e. *GNU Make*). En entornos GNU/Linux es fácilmente instalable a través del sistema de paquetes.

### Los puntos de montaje
Si bien todas las dependencias software están encapsuladas en el contenedor, los scripts para manejarlo se encargan de montar automáticamente una serie de directorios dentro del contenedor, permitiendo que éste lea y modifique datos cuya persistencia no está ligada a la existencia del contenedor.

Por defecto el contenedor contará con los siguientes directorios procedentes del host:
* `$PWD/notebook`, montado en `/home/jovyan/work`, contiene los notebooks que serán creados y/o manejados por el contenedor Jupyter.
* `$PWD/.jupyter`, montado en `/home/jovyan/.jupyter`, contiene la configuración interna del notebook.

### Configuración del contenedor
Las características específicas del contenedor se definen en las variables de entorno definidas en el fichero `jupyterlab.env`. El fichero incluye una configuración por defecto *sensible enough*, pero permite cambiar fácilmente aspectos del contenedor:
* `DOCKER_IMAGE`, imagen Docker a usar
* `DOCKER_CONTAINER_NAME`, nombre que Docker asignará a la imagen que creemos
* `CONT_PORT`, puerto TCP en el que Jupyter estará escuchando dentro del contenedor
* `HOST_PORT`, puerto TCP del host a través del que nos conectaremos a Jupyter, siendo redireccionado automáticamente a `CONT_PORT`

### Crear (y configurar) el contenedor
El contenedor Docker que ejecutará el notebook se crea mediante el siguiente comando:
```
$ make run
```
Este comando leerá las variables de configuración de `jupyterlab.env` y creará un contenedor **persistente**. El comando tomará control de la salida estándar y mostrará los logs generados por Jupyter, entre ellos la **URL de conexión** al notebook una vez el arranque haya finalizado.

Una vez creado, es necesario aplicar una configuración inicial al contenedor. Esta inicialización consiste básicamente en instalar una serie de dependencias que no están incluidas por defecto en la imagen por defecto.
```
$ make config
```
La instalación de dependencias puede tomar cierto tiempo, pero sólo es necesario ejecutar este paso después de crear un nuevo contenedor con `make run`.

### Detener ejecución del contenedor
El contenedor puede ser detenido pulsando `Ctrl + C` en la terminal en la que se ejecuta, o bien ejecutando el comando:
```
$ make stop
```
Todo el contenido y modificaciones realizados en el contenedor seguirán disponibles en caso de que queramos reanudar su ejecución.

### Reanudar ejecución del contenedor
Un contenedor previamente detenido puede ser reanudado con el comando:
```
$ make start
```
La salida estándar mostrará el log de Jupyter, incluida la URL a través de la que podemos volver a conectar con el contenedor.

### Eliminar contenedor
**AVISO:** eliminar el contenedor conlleva el borrado del software, dependencias y configuración de Jupyter pero **no** el del contenido del directorio local `$PWD/notebook`.

El siguiente comando elimina el contenedor:
```
$ make rm
```
