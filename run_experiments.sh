beta="10"
flips="0.00"
repeats="65 129 257 513"
attack="hsj_rep"
num_samples=100
noise="bayesian"
device=0
for flip in $flips; do
  for repeat in $repeats; do
    echo "========================================"
    exp_name="$attack""_r_$repeat""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
    export CUDA_VISIBLE_DEVICES=$device
    echo "Device: $device"
    fixed_params="-d mnist -pf 0.1 -q 5 -ns $num_samples -r $repeat -fp $flip"
    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
    command2="python crunch_experiments.py $exp_name"
    device=$(( (device + 1) % 4 ))
    echo "$command1"
    (nohup $command1; $command2)  > logs/$exp_name.txt &
  done
done