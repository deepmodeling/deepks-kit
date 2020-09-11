nohup python -u -m deepqc iterate args.yaml >> log.iter 2> err.iter &
echo $! > PID
