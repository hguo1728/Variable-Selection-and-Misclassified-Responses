# Project Structure and Description

*Addressing both variable selection and misclassified responses with parametric and semiparametric methods*

Submitted to *Bernoulli*.

This repository contains the implementation and example usage of the proposed method. The structure is as follows:

---

## 📘 `example.ipynb`

This Jupyter notebook includes:

1. **Data Simulation**  
   Generates synthetic data for evaluating the method under controlled conditions.

2. **Method Usage**  
   Demonstrates how to apply the proposed method using the simulated data.  
   Includes:
   - Setting hyperparameters
   - Calling the main function

---

## `run.py`

This script serves as the **entry point** for running the proposed method. It performs:

- **Function Calls**  
  Executes the main training and evaluation logic using functions from the `train/` directory.

- **Argument Descriptions**

---

## `train/`

This folder contains the **core implementation** of the proposed methods:

- Model training and optimization
- Supporting utilities 

