#!/bin/bash
cd $1
#yes
#echo $anchor
anchor=$(./darknet detector map $2 $3 $4)
echo $anchor
