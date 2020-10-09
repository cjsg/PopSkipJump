betas="1 5 10 50"
attacks="psj"
num_samples=100
noise="bayesian"
for attack in $attacks; do
  for beta in $betas; do
    echo "========================================"
    exp_name="$attack""_b_$beta""_$noise""_ns_$num_samples"
    command="python app.py -d mnist -o $exp_name -n $noise -a $attack -ns $num_samples -b $beta"
    echo $command
    $command
  done
done