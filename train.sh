#!/bin/bash
cd $1
#yes
#echo $anchor
anchor=$(./darknet detector calc_anchors $2 -num_of_clusters $3 -width $4 -height $5)