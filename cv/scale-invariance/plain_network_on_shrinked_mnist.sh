for ((n_plain_layers=11;n_plain_layers!=21;n_plain_layers++))
do
  for gpu_index in 0 1 2 3
  do
    postfix=round-$((gpu_index + $1 * 4))
    echo $postfix
    ipython plain_network_on_shrinked_mnist.py -- \
      --gpu_index=$gpu_index --n_plain_layers=$n_plain_layers --postfix=$postfix &
  done
  wait
done
