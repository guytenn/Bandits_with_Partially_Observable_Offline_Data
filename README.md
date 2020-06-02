# Bandits with Partially Observable Offline Data

Example of how to run:
python3.8 run.py --algo square --N 5 --d 30 --K 50 --l 0.1 --delta 0.01 --L_values 0 5 10 15 20 25 --gamma_values 1 5 10 15 --seed 10 --max_jobs 40 --verbose --x_normalization 10 --noise bernoulli

Run with partially observable offline dataset:
python3.8 run.py --N 6 --d 30 --K 30 --delta 0.01 --L_values 0 5 10 15 20 25 --alpha_l_values 0.01 0.1 0.3 0.7 1 --gamma_factor 0.1 --max_jobs 40 --sigma 1 --x_normalization 1 --perturbations --data_size 1000000 --verbose --seed 10