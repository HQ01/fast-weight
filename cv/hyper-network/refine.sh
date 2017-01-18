for ((N=0; N!=21; N++))
do
  echo N $N
  ipython refine.py $N
done
