# FLORENCEV2
## BUILD IT
 <code> sudo docker buildx build --platform=linux/amd64 -t your\_docker\_hub\_user\_name/florencev2 -f Dockerfile --push . </code> 
## PULL IT
 <code> sudo docker pull your\_docker\_hub\_user\_name/florencev2:latest </code>
## RUN IT
 <code> sudo docker run --gpus all -it --rm -v /path/to/florencev2/code:/florencev2 your\_docker\_hub\_user\_name/florencev2:latest </code>
