beta="1"
flip="0.00"
repeat="1"
attack="hsj"
num_samples=20
noise="deterministic"
dataset="mnist"
isc="4"
ets="pca"
etd="50"
device=0
anes="1"
#for etd in $etds; do
#  for attack in $attacks; do
#    exp_name="$dataset""_$attack""_isc_$isc""_et_$et""_etd_$etd""_r_$repeat""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
#    export CUDA_VISIBLE_DEVICES=$device
#    echo "Device: $device"
#    fixed_params="-d $dataset -pf 0.1 -q 1 -ns $num_samples -r $repeat -fp $flip -isc $isc -et $et -etd $etd"
#    command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params"
#    command2="python crunch_experiments.py $exp_name $dataset"
#    device=$(( (device + 1) % 4 ))
#    echo "$command1"
#    (nohup $command1; $command2)  > logs/$exp_name.txt &
##      $command2
#  done
#done
for id in `seq 1 10`
do
  for ane in $anes; do
    for et in $ets; do
      exp_name="$dataset""_$ane""_$id""_$attack""_isc_$isc""_et_$et""_etd_$etd""_r_$repeat""_b_$beta""_$noise""_fp_$flip""_ns_$num_samples"
      export CUDA_VISIBLE_DEVICES=$device
      echo "Device: $device"
      fixed_params="-d $dataset -ns $num_samples -r $repeat -fp $flip -isc $isc -et $et -etd $etd"
      command1="python app.py -o $exp_name -n $noise -a $attack -b $beta $fixed_params -ane $ane"
      command2="python crunch_experiments.py $exp_name $dataset"
      device=$(( (device + 1) % 4 ))
      echo "$command1"
#      (nohup $command1; $command2)  > logs/$exp_name.txt &
      $command2 &
    done
  done
done


