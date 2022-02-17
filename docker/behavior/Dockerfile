ARG REGISTRY=docker.io
ARG REPO=igibson
ARG VERSION=v2.0.6

FROM $REGISTRY/$REPO:$VERSION 

RUN git clone --depth 1 --branch master https://github.com/StanfordVL/bddl /opt/bddl --recursive
RUN pip install --no-cache-dir -e /opt/bddl
