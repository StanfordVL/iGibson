#!/bin/bash

BASE_DIR=$1

for FNAME in "$BASE_DIR"/*
do
  BASENAME="$(basename $FNAME)"
  if [[ $BASENAME == *.blend ]]
  then
    OBJECT_ID=${BASENAME:0:-6}
    CATEGORY=${OBJECT_ID:0:-4}
    echo "File name: "$FNAME
    echo "Category: "$CATEGORY
    echo "Object ID: "$OBJECT_ID
    ./process_object_new.sh $FNAME $CATEGORY $OBJECT_ID
  fi
done