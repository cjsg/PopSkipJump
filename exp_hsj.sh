betas="1 5 10 50"
attacks="psj"
num_samples=100
noise="bayesian"
device=0
for attack in $attacks; do
  for beta in $betas; do
    echo "========================================"
    exp_name="$attack""_b_$beta""_$noise""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d mnist -pf 0.1 -q 5 -ns $num_samples"
    command="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    echo "Command: $command"
    device=$(( device + 1 ))
    nohup $command > logs/$exp_name.txt &
  done
done