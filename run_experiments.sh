betas="1 2 5 10 20 50 100"
flip=0.05
repeat=1
attacks="psj"
num_samples=100
noise="bayesian"
device=0
for beta in $betas; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$attack""_r_$repeat""_b_$beta""_$noise""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d mnist -pf 0.1 -q 5 -ns $num_samples -r $repeat -fp $flip"
    command="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    echo "Command: $command"
    device=$(( (device + 1) % 4 ))
    nohup $command > logs/$exp_name.txt &
  done
done