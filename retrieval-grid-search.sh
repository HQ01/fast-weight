# learning rate
for lr in 0.03 0.02 0.01 0.008
do
  ipython retrieval.py $lr &
done
wait

# inner loop
