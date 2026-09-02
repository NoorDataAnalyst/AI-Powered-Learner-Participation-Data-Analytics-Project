# AI-Powered Learner Participation & Behavioral Analytics Framework

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

> An empirical analytics and machine learning framework for understanding, predicting, and optimizing learner participation and conversion across digital educational platforms.

**Authors:** Inshaal Tariq, Noor ul Ain Zahid, Suchana Panja, Adeniji Shukrat Opeyemi, Simphiwe Mdakane, Gideon Asante Yeboah, Shivanshi Agrawal, Oladeji Faith, Tayyaba Munir, Soumya Ranjan Jena, Chabi Raleting
**Affiliation:** Excelerate AI-Powered Data Analytics Internship Cohort — Team 02
**Date:** March 2026

---

## 📑 Table of Contents

* [1. Executive Summary](#1-executive-summary)
* [2. Introduction & Background](#2-introduction--background)
* [3. Dataset & Data Architecture](#3-dataset--data-architecture)
* [4. Analytical Approach & Technical Framework](#4-analytical-approach--technical-framework)
* [5. Key Findings & Exploratory Analysis](#5-key-findings--exploratory-analysis)
* [6. Model Performance & Baseline Experimental Results](#6-model-performance--baseline-experimental-results)
* [7. Strategic Recommendations & Actionable Insights](#7-strategic-recommendations--actionable-insights)
* [8. Limitations & Known Risks](#8-limitations--known-risks)
* [9. Conclusion & Future Work](#9-conclusion--future-work)
* [10. Repository Architecture & Installation](#10-repository-architecture--installation)
* [License](#-license)

---

## 1. Executive Summary

This project addresses the operational challenge of understanding, predicting, and optimizing learner conversion and participation dynamics across digital educational platforms.

Using an empirical dataset of **8,558 learner records** across **23 distinct learning opportunities**, we developed a scalable analytical framework to:

* Map learner engagement trajectories
* Identify participation and drop-off drivers
* Engineer behavioral and temporal features
* Extract predictive signals for machine learning classification
* Generate actionable recommendations for improving learner conversion and retention

### Key Findings

The analysis demonstrates a significant divergence between application volume and active participation:

* **Internships** account for **63.3% (5,421)** of total applications but have an active participation rate of only **21.5%**, indicating substantial selection friction.
* Non-internship categories — **Courses, Events, Engagements, and Competitions** — achieve substantially higher conversion rates ranging from **84.2% to 96.9%**.
* Learners who apply **1–3 months after registration** demonstrate a **56.8% active conversion rate**, compared with **47.5% for same-day applicants**.
* Feature engineering indicates that **onboarding velocity** and **opportunity category** provide valuable signals for predicting learner participation.

These findings support targeted intervention strategies, improved onboarding, and personalized opportunity recommendations.

---

## 2. Introduction & Background

### Domain Context

EdTech and digital career platforms often experience significant funnel leakage between:

**Registration → Application → Selection → Participation → Completion**

While platform growth metrics frequently emphasize the number of registered users, sustainable platform performance depends on active participation, successful completion, and repeat engagement.

### Problem Statement

Platform administrators lack sufficient visibility into the behavioral variables that differentiate high-intent learners from casual applicants and eventual drop-outs.

Without identifying predictive signals early in the onboarding lifecycle, retention strategies tend to remain **reactive rather than proactive**.

### Primary Objectives & Research Questions

1. **Conversion Mapping**
   What structural factors govern learner progression across different opportunity types?

2. **Temporal Dynamics**
   How does onboarding velocity (`days_to_apply`) affect active participation success?

3. **Behavioral Segmentation**
   Can engineered behavioral features reliably isolate high-churn cohorts before program kickoff?

---

## 3. Dataset & Data Architecture

### Data Sources & Schema

The primary analytical dataset consists of:

| Attribute                           |     Value |
| ----------------------------------- | --------: |
| Learner/Application Records         | **8,558** |
| Learning Opportunities              |    **23** |
| Feature Dimensions After Processing |    **34** |

The dataset contains demographic information, timestamps, opportunity categories, application information, participation status, and engineered behavioral features.

### Data Processing Pipeline

```text
+-----------------------------------------------------------------------------------+
|                              RAW LEARNER DATA                                    |
|                                                                                   |
|        8,558 Records | Demographics | Timestamps | Opportunity Types             |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                         PRE-PROCESSING & VALIDATION                              |
|                                                                                   |
|  • Regex String Normalization       • Missing Value Imputation                   |
|  • Typo Correction & Deduplication  • ISO 8601 Date Parsing                     |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                         FEATURE ENGINEERING LAYER                                |
|                                                                                   |
|  • Target Isolation                                                            |
|    (`is_active_participation`, `is_dropped_or_rejected`)                       |
|                                                                                   |
|  • Temporal Features                                                            |
|    (`days_to_apply`, `engagement_duration_days`)                               |
|                                                                                   |
|  • Behavioral Aggregates                                                       |
|    (`participation_frequency`, `is_repeat_participant`)                        |
+-----------------------------------------------------------------------------------+
```

### Feature Engineering Dictionary

| Feature                    | Type            | Operationalization Logic                                 | Analytical Objective                          |
| -------------------------- | --------------- | -------------------------------------------------------- | --------------------------------------------- |
| `engagement_duration_days` | Continuous      | `End Date - Start Date`                                  | Quantifies program duration commitment        |
| `days_to_apply`            | Continuous      | `Apply Date - Signup Date`                               | Measures user onboarding velocity             |
| `participation_frequency`  | Discrete        | Sum of opportunities applied per user                    | Measures platform activation depth            |
| `is_repeat_participant`    | Binary          | `1` if frequency > 1, otherwise `0`                      | Segments baseline users vs. power users       |
| `is_active_participation`  | Target — Binary | `1` if status ∈ {Started, Team Allocated, Rewards Award} | Classification target for positive engagement |
| `is_dropped_or_rejected`   | Target — Binary | `1` if status ∈ {Withdrawn, Dropped, Rejected}           | Classification target for negative engagement |

---

## 4. Analytical Approach & Technical Framework

### Mathematical Formulations

#### 4.1 Onboarding Velocity Index (`Vₒ`)

To model user activation delay, the non-negative interval between registration and application is calculated as:

$$
V_o = \max\left(0,\frac{T_{\text{apply}}-T_{\text{signup}}}{86,400}\right)
$$

Where:

* $T_{\text{apply}}$ = application timestamp
* $T_{\text{signup}}$ = registration timestamp
* $86,400$ = number of seconds in one day

The timestamps are represented as POSIX timestamps in seconds.

#### 4.2 Category Participation Density (`Dᶜ`)

The density of active participation relative to applications is calculated as:

$$
D_c = \frac{N_{\text{active}}}{N_{\text{applied}}}\times100
$$

Where:

* $N_{\text{active}}$ = number of actively participating records
* $N_{\text{applied}}$ = total number of applications

---

## 5. Key Findings & Exploratory Analysis

### 5.1 Empirical Summary Statistics

| Metric                                            | Sample (n) |   Mean | Median | Std. Dev. | IQR (25%–75%) |
| ------------------------------------------------- | ---------: | -----: | -----: | --------: | ------------: |
| Learner Age (`age`)                               |      8,558 |  23.75 |  23.00 |      4.37 |   22.00–25.00 |
| Application Lag (`days_to_apply`)                 |      8,024 |  57.65 |   5.00 |     89.08 |    1.00–93.00 |
| Program Length (`engagement_duration_days`)       |      3,473 | 373.04 | 468.00 |    290.68 |  63.00–604.00 |
| Application Frequency (`participation_frequency`) |      8,558 |   5.76 |   4.00 |      5.75 |     1.00–9.00 |

### 5.2 Categorical Funnel Breakdown

| Opportunity Type | Application Volume | Active Participation |
| ---------------- | -----------------: | -------------------: |
| **Internship**   |          **63.3%** |            **21.5%** |
| Course           |               9.1% |                96.9% |
| Event            |               8.2% |                94.1% |
| Competition      |               6.8% |                84.2% |
| Engagement       |               5.4% |                91.2% |

### 5.3 Major Insights

#### 1. Internship Bottleneck

Internships represent **63.3% (5,421)** of total platform demand but have an active participation rate of only **21.5%**.

This indicates significant friction within the internship application and selection funnel, including a **65%+ rejection rate**.

#### 2. Onboarding Sweet Spot

Immediate applicants — those applying within **0–1 days** — demonstrate higher churn than learners who spend approximately **30–90 days** browsing the opportunity catalog before applying.

The latter group achieves an active conversion rate of approximately **56.8%**.

---

## 6. Model Performance & Baseline Experimental Results

Preliminary classification models were trained to predict:

`is_active_participation`

The experimental setup used:

* **80/20 train-test split**
* **5-fold cross-validation**
* Standardized features
* Binary classification

### Model Performance

| Model Architecture             | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------------------------ | --------: | --------: | --------: | --------: |
| Logistic Regression — Baseline |     0.712 |     0.684 |     0.698 |     0.745 |
| Random Forest Classifier       |     0.834 |     0.812 |     0.823 |     0.889 |
| **XGBoost Gradient Boosting**  | **0.861** | **0.845** | **0.853** | **0.912** |

### Best Performing Model

**XGBoost Gradient Boosting** achieved the strongest baseline performance across all reported metrics:

* **Precision:** 0.861
* **Recall:** 0.845
* **F1-Score:** 0.853
* **ROC-AUC:** 0.912

Based on these preliminary experiments, XGBoost is the leading candidate for future productionization.

---

## 7. Strategic Recommendations & Actionable Insights

### Short-Term Priorities — 0–3 Months

#### Re-engage Warm Leads

Target registered users around the **30-day mark** with curated non-internship opportunities, particularly:

* Courses
* Events
* Other high-conversion opportunities

This strategy can leverage the identified higher-conversion onboarding window.

#### Internship Expectation Management

Introduce preliminary screening and expectation-setting mechanisms for internship applicants to:

* Reduce application friction
* Improve applicant understanding of selection processes
* Reduce avoidable drop-offs

### Mid-Term Priorities — 3–6 Months

#### Automated Churn Warnings

Develop early-warning pipelines using engineered behavioral signals such as:

* `days_to_apply`
* `participation_frequency`
* `is_repeat_participant`

These signals can be used to trigger automated retention workflows.

### Long-Term Priorities — 6–12 Months

#### Dynamic Recommendation Engine

Develop a personalized opportunity recommendation system based on:

* Historical participation
* Opportunity category
* Completion likelihood
* Learner behavioral patterns

The objective is to personalize opportunity feeds based on **predicted completion likelihood rather than application volume alone**.

---

## 8. Limitations & Known Risks

### Temporal Censoring

Right-censoring exists for long-duration programs where:

`engagement_duration_days > 365`

Some completion outcomes may still be ongoing and therefore unavailable at the time of analysis.

### Missing Value Heterogeneity

Start and end dates are primarily populated for structured courses. As a result, approximately **59.4% of duration metrics are null** for open-ended engagements.

### Selection Bias

The high active participation rates observed for Events and Courses may partly reflect their lower barriers to entry rather than genuinely higher overall learner retention.

Therefore, category-level conversion rates should be interpreted within the context of the underlying application and selection mechanisms.

---

## 9. Conclusion & Future Work

This analytical framework demonstrates that **onboarding velocity** and **opportunity classification** provide strong structural signals for understanding learner participation and retention.

The project establishes a foundation for moving from descriptive analytics toward predictive and prescriptive learner engagement strategies.

### Future Development

1. **Productionize the XGBoost Classification Model**
   Deploy the trained model as a microservice API for real-time or batch prediction.

2. **Integrate Survival Analysis**
   Extend the framework beyond binary participation prediction to estimate the timing and probability of learner drop-off.

3. **Develop Personalized Recommendations**
   Use behavioral and predictive signals to recommend opportunities with a higher likelihood of successful participation.

---

## 10. Repository Architecture & Installation

### Repository Structure

```text
.
├── data/
│   ├── raw_learner_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning_and_qc.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_exploratory_data_analysis.ipynb
│
├── scripts/
│   ├── full_analysis.py
│   └── train_models.py
│
├── reports/
│   └── executive_summary.pdf
│
├── requirements.txt
├── LICENSE
└── README.md
```

### Directory Description

| Directory/File                                 | Purpose                                                        |
| ---------------------------------------------- | -------------------------------------------------------------- |
| `data/raw_learner_data.csv`                    | Raw learner dataset containing 8,558 records                   |
| `data/processed_data.csv`                      | Processed dataset with engineered features                     |
| `notebooks/01_data_cleaning_and_qc.ipynb`      | Data quality control and missing-value handling                |
| `notebooks/02_feature_engineering.ipynb`       | Feature extraction and transformation                          |
| `notebooks/03_exploratory_data_analysis.ipynb` | Exploratory analysis, visualization, and statistical profiling |
| `scripts/full_analysis.py`                     | End-to-end analytical processing pipeline                      |
| `scripts/train_models.py`                      | Machine learning model training                                |
| `reports/executive_summary.pdf`                | Executive summary of project findings                          |
| `requirements.txt`                             | Python dependencies                                            |
| `LICENSE`                                      | MIT License                                                    |
| `README.md`                                    | Project documentation                                          |

---

## Installation & Execution Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/learner-analytics-framework.git
cd learner-analytics-framework
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Upgrade `pip` and install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute the Data Pipeline

Run the complete analytical pipeline:

```bash
python scripts/full_analysis.py
```

### 5. Train the Machine Learning Models

If the model-training script is available in the repository:

```bash
python scripts/train_models.py
```

---

## 🧪 Reproducibility

To reproduce the analysis:

1. Clone the repository.
2. Create and activate the Python virtual environment.
3. Install dependencies using `requirements.txt`.
4. Run the data-processing pipeline.
5. Execute the exploratory analysis notebooks.
6. Run the model-training script.
7. Review the generated reports and model performance metrics.

---

## 📊 Key Project Metrics

| Metric                               |      Result |
| ------------------------------------ | ----------: |
| Learner Records                      |   **8,558** |
| Learning Opportunities               |      **23** |
| Feature Dimensions                   |      **34** |
| Internship Application Share         |   **63.3%** |
| Internship Active Participation      |   **21.5%** |
| Highest Reported Category Conversion |   **96.9%** |
| Best Model                           | **XGBoost** |
| Best Precision                       |   **0.861** |
| Best Recall                          |   **0.845** |
| Best F1-Score                        |   **0.853** |
| Best ROC-AUC                         |   **0.912** |

---

## 📄 License

This repository is distributed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the complete license terms.

---

## 👥 Team

**Excelerate AI-Powered Data Analytics Internship — Team 02**

* Inshaal Tariq
* Noor ul Ain Zahid
* Suchana Panja
* Adeniji Shukrat Opeyemi
* Simphiwe Mdakane
* Gideon Asante Yeboah
* Shivanshi Agrawal
* Oladeji Faith
* Tayyaba Munir
* Soumya Ranjan Jena
* Chabi Raleting
