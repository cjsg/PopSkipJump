beta="1"
flip="0.00"
repeat="1"
attacks="psj hsj"
num_samples=10
dm="l2"
noise="deterministic"
sn="0.01"
cs="26"
drs="0.5"
dataset="mnist"
device=0
for dr in $drs; do
  for attack in $attacks; do
    echo "========================================"
    exp_name="$dataset""_$attack""_r_$repeat""_sn_$sn""_cs_$cs""_dr_$dr""_dm_$dm""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d $dataset -ns $num_samples -r $repeat -fp $flip -dm $dm -sn $sn -cs $cs -dr $dr"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name $dataset"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
#    (nohup $command1; $command2)  > logs/$exp_name.txt &
    $command1
    $command2
  done
done


