#!/bin/bash

rm -f e_*raw mo_*raw dist.raw
for jj in data_*_*; 
do 
    for ii in e_hf.raw  e_mp2.raw  mo_coeff.raw mo_ener.raw dist.raw
    do 
	cat $jj/$ii >> $ii; 
    done
done

cat $jj/system.raw > system.raw
