nohup python -u -m deepks iterate machines.yaml params.yaml systems.yaml scf_abacus.yaml init_scf_abacus >> log.iter 2> err.iter & 
echo $! > PID