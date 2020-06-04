# Experiments in FAT ML using the Bank Marketing Data Set of Moro et al.

## Set up

1. Install ananconda and start a commond prompt for it
1. Create a conda environment for it
    1. `conda env create -f fat_ml.yml`
    * If you get the error msg. below, change the name in the first line of the file from `fat_ml` to `fat_ml_test` (or something else besides fat_ml).  In the next step, `fat_ml` will become `fat_ml_test`.    
	   `Error msg.: CondaValueError: prefix already exists: <...>\anaconda3\envs\fat_ml`
	1. `conda activate fat_ml`
1. Test the install
    1.  `cd src`
	1.  `python Logistic "Regresion balanced.py"`
	1.  Check the output matches what's in Logistic Regression balanced_log.txt (look at RFE, the 1st fit, the 2nd fit and the last fit)