#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 30
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/shaunak/miniconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/shaunak/miniconda2/etc/profile.d/conda.sh" ]; then
        . "/home/shaunak/miniconda2/etc/profile.d/conda.sh"
    else
        export PATH="/home/shaunak/miniconda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ML
for i in {0..3}
do
	python sums_train.py $i
	echo "Done with $i"
done

echo "Finished"
