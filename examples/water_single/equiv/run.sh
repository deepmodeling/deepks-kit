nohup python -u -m deepks iterate args.yaml >> log.iter 2> err.iter &
echo $! > PID
