beta=1
flips="0.05 0.10"
attacks="hsj_rep psj"
num_samples=100
noise="stochastic"
device=0
for flip in $flips; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$attack""_b_$beta""_$noise""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d mnist -pf 0.1 -q 5 -ns $num_samples -r 90 -pf $flip"
    command="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    echo "Command: $command"
    device=$(( device + 1 ))
    nohup $command > logs/$exp_name.txt &
  done
done