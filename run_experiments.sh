betas="1"
flip="0.00"
repeat="1"
attacks="psj"
num_samples=100
noise="deterministic"
dataset="cifar10"
device=2
for beta in $betas; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$dataset""_$attack""_r_$repeat""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d $dataset -pf 0.1 -q 5 -ns $num_samples -r $repeat -fp $flip"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name $dataset"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
    (nohup $command1; $command2)  > logs/$exp_name.txt &
  done
done


