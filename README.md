# 636-Deep-Learning-Project
## Title
Deep Neural Network for m-Height Estimation

## Main Notebook
👉 [Start here: Project3.ipynb](Project3/Project3.ipynb)
## Research Questions / Problem Definition
Can we efficiently estimate the m-height of an analog code using deep learning, instead of solving expensive optimization problems?  

**Background** LP-based approach  

The classical method formulates the m-height problem as a collection of linear programming (LP) problems.
For a given generator matrix G, the algorithm enumerates combinations of indices and solves multiple LPs to find the maximum objective value.
Pros: exact
Cons: computationally expensive and scales poorly as problem size grows.

**Input**
Each input consists of:
- Three integers (n, k, m)
- A matrix P, where G = [I | P]

In this project, we focus on:
- n = 9
- k ∈ {4, 5, 6}
- m ∈ {2, ..., n − k}

**Target**
Learn the function using NN  
$f(n, k, m, P) \rightarrow h_m(C)$
## Method
Different input representations were explored, including:
- Flattening the P-matrix into a vector (for MLP-based models)
- Treating rows as tokens for Transformer-based models
- Convolutional representations (CNN), which were found to be less effective
### Feature Engineering
Additional handcrafted features were extracted from the P-matrix to provide structural signals and improve model performance

### Model Architectures
Several architectures were evaluated:
- **MLP (Dense network):** captures global relationships between features
- **CNN:** tested for spatial pattern extraction, but performed poorly due to lack of spatial structure in the data
- **Transformer:** used to model row-wise relationships in the P-matrix

A **hybrid model** was developed by combining MLP and Transformer outputs through a stacked architecture, followed by a small Dense head.
### Optimization
Models were trained using standard regression objectives (log-MSE), with hyperparameters tuned across:
- Network depth and width
- Learning rate
- Dropout
- Optimizers
  
## Data

- Synthetic dataset generated using an LP-based algorithm (used as ground truth)
- Provided dataset with fixed parameter settings (n = 9, k ∈ {4,5,6}, m ∈ {2,...,n−k})

The P-matrix serves as the core input. Multiple representations were explored, including flattening and convolution-based approaches.
   
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
```
.
├── Project1/
│ └── Project.ipynb
│
├── Project2/
│ ├── project2.ipynb
│ ├── DS-2-Train-mHeights
│ └── DS-2-Train-n_k_m_P
│
├── Project3/
│ ├── Project3.ipynb
│ ├── DS-3-Train-mHeights
│ └── DS-3-Train-n_k_m_P
│
└── README.md
```

