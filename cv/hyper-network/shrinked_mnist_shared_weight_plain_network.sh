for ((n_plain_layers=11;n_plain_layers!=21;n_plain_layers++))
do
  for gpu_index in 0 1 2 3
  do
    postfix=round-$((gpu_index + $1 * 4))
    echo $postfix
    ipython shrinked_mnist_shared_weight_plain_network.py -- \
      --gpu_index=$gpu_index --n_plain_layers=$n_plain_layers --postfix=$postfix &
  done
  wait
done
