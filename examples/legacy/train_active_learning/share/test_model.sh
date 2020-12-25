for fd in model.*; do mkdir $fd/test; done

echo training set
python /path/to/source/deepqc/train/test.py -m model*/model.pth -d `cat train_paths.raw` -o test/train -D dm_eig se_eig fe_eig

echo testing set
python /path/to/source/deepqc/train/test.py -m model*/model.pth -d `cat test_paths.raw` -o test/test -D dm_eig se_eig fe_eig
