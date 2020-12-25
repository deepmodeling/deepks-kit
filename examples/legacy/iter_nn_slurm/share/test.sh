mkdir test
python /path/to/source/deepqc/train/test.py -m model.pth -d `cat train_paths.raw` -o test/train
python /path/to/source/deepqc/train/test.py -m model.pth -d `cat test_paths.raw` -o test/test
