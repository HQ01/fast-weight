for INTERVALS in '1 2 4 8' '2 4 8' '1 4 8' '1 2 4' '1 2' '1 4' '1 8' '2 4' '2 8' '4 8' '1' '2' '4' '8'
do
  echo $INTERVALS
  ipython train_hybrid_weight_residual_network.py 8 $INTERVALS
done
