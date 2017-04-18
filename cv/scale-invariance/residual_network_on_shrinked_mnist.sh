for ((n_residual_layers=1;n_residual_layers!=21;n_residual_layers++))
do
  for gpu_index in 0 1 2 3
  do
    postfix=round-$((gpu_index + $1 * 4))
    echo $postfix
    ipython residual_network_on_shrinked_mnist.py -- \
      --gpu_index=$gpu_index --n_residual_layers=$n_residual_layers --postfix=$postfix &
  done
  wait
done
