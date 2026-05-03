# 636-Deep-Learning-Project
## Title
Deep Neural Network for m-Height Estimation

## Motivation / Why it matters
Main Notebook (Start Here)
## Research Questions / Problem Definition
Can we efficiently estimate the m-height of an analog code using deep learning, instead of solving expensive optimization problems?  

**Background** LP-based approach  

The classical method formulates the m-height problem as a collection of linear programming (LP) problems.
For a given generator matrix G, the algorithm enumerates combinations of indices and solves multiple LPs to find the maximum objective value.
Pros: exact
Cons: computationally expensive and scales poorly as problem size grows.

**Input**
Code parameters: $n, k, m$
Generator matrix $G = [I_k \mid P]$, where $P \in \mathbb{R}^{k \times (n-k)}$
For every DNN input, 3 integers (n=9,m= {2,3,...,n-k},k = {4,5,6}) and matrix P (G = (I|P))are given.

**Target**
Learn the function using NN  
$f(n, k, m, P) \rightarrow h_m(C)$
## Method
## Data
1. synthetic dataset
2. professor provided (n=9, k= , m =.., P)
   P is a matrix, I explored different representations of P, including flattening (vectorization) and convolution-based approaches using CNNs.
   
Project 1 – Project 3 represent the iterative development of the model, including different attempts and improvements over time.

The size of the test set gradually increased across the three stages: 32,568 samples in Project1, 44,388 in Project2, and 56,363 in Project3.

In Project2 and Project3, I augmented the training data by generating additional samples using the LP-based algorithm as ground truth. This helped improve the model’s performance.
   
## Results / Key Findings
- Built a deep learning pipeline to approximate the m-height function from structured matrix inputs.
- Demonstrated that **CNNs are not suitable**(**log-MSE ≈ 1.4**) for P-matrix data due to lack of spatial locality, while **MLPs better capture global relationships**(**log-MSE ≈ 1.2**).
- Proposed a **stacked hybrid model (Dense + Transformer)** to improve prediction performance.
- Enhanced input representation via **manual feature engineering** on P-matrix patterns.
- Expanded training data using an **LP-based algorithm**, leading to more stable and accurate predictions.
- Final model achieved **log-MSE ≈ 0.65**.
## Dependencies
- Python
- numpy
- pandas
- scikit-learn
- tensorflow / keras  
- matplotlib
  
## Repo Structure

