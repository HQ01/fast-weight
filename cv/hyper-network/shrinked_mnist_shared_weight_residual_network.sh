for ((n_residual_layers=11;n_residual_layers!=21;n_residual_layers++))
do
  for gpu_index in 0 1 2 3
  do
    postfix=round-$((gpu_index + $1 * 4))
    echo $postfix
    ipython shrinked_mnist_residual_network.py -- \
      --gpu_index=$gpu_index --n_residual_layers=$n_residual_layers --postfix=$postfix &
  done
  wait
done
