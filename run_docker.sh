#!/bin/bash
echo "Delete output.csv"
rm -rf output.csv
echo "Compile dockerfile"
docker build -t car_purchase_pred .
echo "Run image and store the file output.csv in the local storage with the current predictions"
docker run -v $(pwd):/app/ -it car_purchase_pred