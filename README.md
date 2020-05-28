# Contextual_Bandits_with_Partially_Observable_Off_Policy_Data

Example of how to run:
python3.8 run.py --algo square --N 5 --d 30 --K 50 --l 0.1 --delta 0.01 --L_values 0 5 10 15 20 25 --gamma_values 1 5 10 15 --seed 10 --max_jobs 40 --verbose --x_normalization 10 --noise bernouli

Run with partially observable offline dataset:
python3.8 run.py --algo square --N 5 --d 30 --K 50 --l 0.1 --delta 0.01 --L_values 0 5 10 15 20 25 --gamma_values 1 5 10 15 --seed 10 --max_jobs 40 --verbose --x_normalization 10 --noise bernouli --perturbations --data_size 1000000