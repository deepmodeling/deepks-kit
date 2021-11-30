nohup python -u -m deepks iterate ../iter/args.yaml orbital.yaml >> log.iter 2> err.iter &
echo $! > PID
