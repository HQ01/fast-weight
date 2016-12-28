# learning rate
lr=0.1
for ((i=0; i!=3; i++))
do
  let lr=lr*0.5 # TODO
  echo $lr
# ipython $1/retrieval.py $lr
done

# inner loop
