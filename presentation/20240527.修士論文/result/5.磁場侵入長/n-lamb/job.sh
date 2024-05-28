#-l nodes=1:ppn=1
ulimit -s unlimited

echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

NUMPRO=1
TOPDIR=/home/matsunaga/anaconda/bin/python3
~/anaconda3/bin/python3 3D.py
