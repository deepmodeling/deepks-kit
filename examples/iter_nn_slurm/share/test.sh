mkdir test
python /home/yixiaoc/SCR/yixiaoc/deep.qc/source_scf/deepqc/train/test.py -m model.pth -d `cat train_paths.raw` -o test/train
python /home/yixiaoc/SCR/yixiaoc/deep.qc/source_scf/deepqc/train/test.py -m model.pth -d `cat test_paths.raw` -o test/test
