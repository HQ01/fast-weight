for decaying_rate in 0 0.5 1 1.5 2 2.5
do
  echo $decaying_rate
  ipython train_cifar_decayed_attention_network.py -- --n_layers=9 --decaying_rate=$decaying_rate --gpu=0,1,2,3
done
