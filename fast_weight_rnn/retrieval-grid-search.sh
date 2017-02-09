for n_hidden_units in 20 50 100
do
  ipython retrieval.py $n_hidden_units 0.005 &
done
wait
