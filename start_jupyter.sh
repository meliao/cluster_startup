#!/bin/bash

export NODEIP=$(hostname -i)
export NODEPORT=$(( $RANDOM + 1024 ))
echo $NODEIP:$NODEPORT
jupyter-notebook --ip=$NODEIP --port=$NODEPORT --no-browser
