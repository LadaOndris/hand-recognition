#!/bin/sh

# Enable recursive **
shopt -s globstar

scp -r ~/projects/IBT/src/**/*.py ladislav_ondris@storage-brno6.metacentrum.cz:/home/ladislav_ondris/IBT/

