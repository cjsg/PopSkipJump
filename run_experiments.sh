beta="1"
flip="0.00"
repeat="1"
attacks="hsj psj"
num_samples=100
noise="smoothing"
sns="0.01 0.005 0.001 0.0005 0.0001"
dataset="mnist"
device=0
for sn in $sns; do
  for attack in $attacks; do
    exp_name="$dataset""_$attack""_r_$repeat""_sn_$sn""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d $dataset -ns $num_samples -r $repeat -fp $flip -sn $sn"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name $dataset"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
    (nohup $command1; $command2)  > logs/$exp_name.txt &
  done
done


