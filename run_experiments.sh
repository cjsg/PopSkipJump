betas="1"
flip="0.00"
repeat="1"
attacks="psj"
num_samples=10
dm="l2"
noise="bayesian"
dataset="mnist"
device=2
for beta in $betas; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$dataset""_$attack""_r_$repeat""_targeted""_dm_$dm""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d $dataset -ns $num_samples -r $repeat -fp $flip -dm $dm --targeted"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name $dataset"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
    (nohup $command1; $command2)  > logs/$exp_name.txt &
#    $command1
#    $command2
  done
done


