for ((N=1; N!=21; N++))
do
  echo N $N
  ipython train_residual_network.py $N
done
