# Experiments in FAT ML using the Bank Marketing Data Set of Moro et al.

## Set up

1. Install ananconda and start a commond prompt for it
1. `conda env create -f fat_ml.yml`
1. Test the install
    1.  `cd src`
	1.  `python Logistic "Regresion balanced.py"`
	1.  Check the output matches what's in Logistic Regression balanced_log.txt (look at RFE, the 1st fit, the 2nd fit and the last fit)