for decaying_rate in 3.0
do
  echo $decaying_rate
  ipython train_cifar_decayed_attention_network.py -- --n_layers=9 --decaying_rate=$decaying_rate --gpu=0,1,2,3
done
