beta="1"
flip="0.00"
repeat="1"
attacks="hsj_all_grad"
num_samples=50
dm="l2"
efs="1"
noise="deterministic"
dataset="mnist"
device=2
for ef in $efs; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$dataset""_$attack""_grad2-rep_r_$repeat""_ef_$ef""_dm_$dm""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d $dataset -ns $num_samples -r $repeat -fp $flip -dm $dm -ef $ef"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name $dataset"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
#    (nohup $command1; $command2)  > logs/$exp_name.txt &
    $command1
    $command2
  done
done


