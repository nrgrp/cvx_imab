# cvx_imab

Code accompanies the paper "Solving Inverse Problem for Multi-armed Bandits via Convex Optimization".

# Abstract
We consider the inverse problem of multi-armed bandits (IMAB) that are widely used in neuroscience and psychology research for behavior modelling.
We first show that the IMAB problem is not convex in general, but can be relaxed to a convex problem via variable transformation.
Based on this result, we propose a two-step sequential heuristic for (approximately) solving the IMAB problem.
We discuss a condition where our method provides global solution of the IMAB problem with certificate, as well as approximations to further save computing time.
Numerical experiments indicate that our heuristic method is more robust than directly solving the IMAB problem via repeated local optimization, and can achieve the performance of Monte Carlo methods within a significantly decreased running time.
We provide the implementation of our method based on CVXPY, which allows straightforward application by users not well versed in convex optimization.

# Run the experiments

To install the required dependencies, run: `pip install -r requirements.txt`.

To run the two experiments, first navigate to the `exp1` or `exp2` folder, then follow these steps:

1. Generate data: `python src/generate_data.py`.

2. Obtain ground truth: `python src/baseline.py`.

3. Solve the IMAB problem.

    3.1. Sequential heuristic: `python src/cvx.py`.
    
    3.2. Sequential heuristic with truncated polynomial approximation: `python src/cvx_truc5.py`.
    
    3.3. Direct method: `python src/direct.py`.
    
    3.4. Monte Carlo method: `python src/mc.py`.

4. Follow the jupyter notebook `notebook/generate_figures.ipynb` to visualize the results.
