# OptiMPa: Data-Driven Life Cycle Assessment (LCA) of Reinforced Concrete Structures

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Refactored%20%26%20Released-success.svg)]()

> **📌 Academic Research Open-Source Release (May 2026)**
> *This repository serves as the official open-source release and refactored codebase for my academic research, originally developed and presented as an academic poster in May 2026. As I prepare to transition into graduate studies in Data Science and Business Analytics, I have refactored and published these predictive models to ensure methodological transparency, encourage peer review, and demonstrate the reproducibility of my analytical pipelines.*

## 📖 Project Overview

The construction sector faces an urgent imperative to decarbonize. A critical challenge lies in reducing the substantial carbon footprint of reinforced concrete without compromising structural integrity or economic viability. **OptiMPa** bridges physical civil engineering principles with advanced data analytics to address this challenge. 

This project utilizes Machine Learning to conduct a highly accurate Life Cycle Assessment (LCA) of various concrete mixtures. By leveraging predictive models to evaluate the environmental impact of Pozzolanic materials, the system identifies optimal material ratios that significantly minimize carbon emissions while maintaining required structural parameters.

## 🎯 Core Objectives
- **Emission Optimization:** Quantify and minimize the Global Warming Potential (GWP) of concrete structures through data-driven material selection.
- **Predictive LCA Modeling:** Replace traditional, time-intensive life cycle assessments with rapid, highly accurate machine learning predictions.
- **Domain Integration:** Demonstrate the synergy between physical civil engineering metrics and computational data science.

## 🔬 Data & Scope

The dataset and subsequent models evaluate a spectrum of concrete mixtures, with a specific analytical focus on the environmental and structural impacts of integrating Pozzolanic materials. 

A primary comparative analysis within the codebase evaluates:
- **Mix 2 (Baseline Baseline/Control):** Standard structural mix parameters.
- **Mix 6 (Pozzolanic Optimized):** Advanced mixture exhibiting high variability in Pozzolanic substitution, serving as the primary test subject for our algorithmic emission reduction metrics.

## ⚙️ Methodology

The project follows a standard end-to-end data science lifecycle, tailored for engineering analytics:

1. **Data Ingestion & Preprocessing:** Utilizing `pandas` for cleaning, handling missing values, and scaling raw material inputs and environmental output metrics.
2. **Feature Engineering:** Transforming physical concrete properties (e.g., binder ratios, aggregate mass, Pozzolanic volume) into predictive features.
3. **Predictive Modeling:** Implementing a **Random Forest Regressor** architecture. This ensemble learning method was selected over linear models due to the non-linear interactions between concrete curing agents and their resulting environmental impacts.
4. **Model Evaluation:** Validation using metrics such as RMSE (Root Mean Squared Error) and R² to ensure the model's reliability for real-world construction analytics.

## 🛠 Tech Stack

- **Core Language:** Python
- **Machine Learning:** `scikit-learn` (Random Forest Implementation, Model Validation)
- **Data Manipulation:** `pandas`, `NumPy`
- **Visualization (Notebooks):** `matplotlib`, `seaborn` (for feature importance and emission variance plotting)
- **Environment:** Jupyter Notebooks / Python Scripts

## 📊 Key Findings

- **High-Fidelity Predictions:** The Random Forest model achieved robust predictive accuracy in forecasting the LCA impact of novel concrete mixtures prior to physical testing.
- **Impact of Pozzolanic Materials:** The algorithmic analysis of **Mix 6** quantitatively proved that specific ratios of Pozzolanic materials yield the highest reduction in carbon emissions without statistically significant degradation in required structural yield metrics.
- **Business/Environmental Value:** The pipeline provides a scalable, analytical framework that construction firms can utilize to balance ESG (Environmental, Social, and Governance) goals with material costs.

## 🚀 How to Run

```bash
# Clone the repository
git clone [https://github.com/kaan36875/optimpa.git](https://github.com/kaan36875/optimpa.git)

# Navigate to the project directory
cd optimpa

# Install required dependencies
pip install -r requirements.txt

# Execute the main predictive pipeline
python src/predictive_lca.py
