# COVID-19 in the U.S.: Statistical Analysis and Mortality Prediction

Who was most at risk from COVID-19, and can mortality be predicted from patient records? This project answers both questions using U.S. COVID-19 case surveillance and household survey data.

## Part 1: Statistical analysis (`Final_Project_Stat.ipynb`)

**Exploratory analysis**
- Hospitalizations vs. deaths over time, by month and year
- Hospitalization and death rates by age group, sex and state
- How pre-existing conditions and risk behaviours relate to hospital admission
- Expected employment loss by sector and demographic group
- Household income vs. delayed or missed medical treatment

**Questions answered with hypothesis tests**
- Are hospitalized patients with underlying conditions more likely to die?
- Which demographic segment is most at risk of death?
- Are asymptomatic patients less likely to be hospitalized or to die?

**Methods:** chi-square tests of independence, one-way ANOVA, a one-sided two-proportion z-test, and OLS regression (with intercept, without intercept, higher-order terms and outlier-cleaned variants).

## Part 2: Mortality prediction (`ML_Task5.ipynb`)

A scikit-learn pipeline predicts whether a patient died (`death_yn`):

- Removed duplicates and non-informative labels ("Unknown", "Missing")
- Undersampled the majority class to address heavy class imbalance
- Most-frequent imputation and one-hot encoding in a `ColumnTransformer`
- `XGBClassifier` with an 80/20 train/test split

| Metric | Score |
|---|---|
| Accuracy | 0.98 |
| Precision | 0.84 |
| Recall | 0.42 |
| F1 | 0.56 |

High accuracy here mostly reflects the class imbalance. Recall of 0.42 means the model misses more than half of deaths, so improving minority-class recall (class weights, threshold tuning, SMOTE) is the clear next step.

## Files

| File | Contents |
|---|---|
| `Final_Project_Stat.ipynb` | Exploratory analysis and hypothesis testing |
| `ML_Task5.ipynb` | Mortality prediction pipeline |
| `Statistical_inference_and_Data_Analysis.pdf` | Written report |
| `Analyzing U.S. COVID-19 Data.pptx` | Presentation slides |

## Tech stack

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, Matplotlib, Seaborn

## How to run

The notebooks were written in Google Colab and load the data from Google Drive. To run them locally, download the datasets and update the `data_path` variables at the top of each notebook.
