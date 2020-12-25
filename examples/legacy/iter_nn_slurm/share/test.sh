mkdir test
python /path/to/source/deepks/train/test.py -m model.pth -d `cat train_paths.raw` -o test/train
python /path/to/source/deepks/train/test.py -m model.pth -d `cat test_paths.raw` -o test/test
