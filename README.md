# Bandits with Partially Observable Offline Data

Example of how to run:
python3.8 run.py --algo square --N 5 --d 30 --K 50 --l 0.1 --delta 0.01 --L_values 0 5 10 15 20 25 --gamma_values 1 5 10 15 --seed 10 --max_jobs 40 --verbose --x_normalization 10 --noise bernoulli

Run with partially observable offline dataset:
python3.8 run.py --algo square --N 6 --d 30 --K 30 --l -0.01 --delta 0.01 --L_values 0 5 10 15 20 25 --gamma_values 1 10 20 30 --seed 10 --max_jobs 40 --verbose --x_normalization 10 --noise bernoulli --perturbations --data_size 1000000