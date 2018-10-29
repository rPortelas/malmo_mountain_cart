#!/bin/bash
cd docker/
echo "launching stuff"
for ((i = 10000; i < $((10000 + $1)); i++)); do
    echo $i
    docker run --net=host malmuu $i &
done
