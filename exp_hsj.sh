beta=1
flip=0.05
attacks="hsj hsj_rep psj"
num_samples=100
noises="stochastic deterministic bayesian"
device=0
for noise in $noises; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$attack""_b_$beta""_$noise""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d mnist -pf 0.1 -q 5 -ns $num_samples -r 90 -fp $flip"
    command="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    echo "Command: $command"
    device=$(( device + 1 ))
    nohup $command > logs/$exp_name.txt &
  done
done