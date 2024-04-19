# Feature Scaling (Normalization) and Feature Encoding

Feature scaling is a data preprocessing technique used to transform the values of features or variables in a dataset to a similar scale. The purpose is to ensure that all features contribute equally to the model and to avoid the domination of features with larger values.
Feature Scaling is the method to limit the range of variables so that they can be compared on common groups.

Feature encoding is a data preprocessing technique used to convert categorical features into numerical features. It is used to convert categorical features into numerical features.

## Data Preprocess

- Detect anomally.
  - Containing special characters
  - Data type
- Detect outliers.
- Detect missing values.
- Detect duplicate.
- Feature Scaling.
  - Small and large unit. So we scale the data to common scale so that we can analyze the data easily.

## Features

1. Let's suppose we have a dataset of containing 4 columns i.e A,B,C and D.
   1. A,B and C will be called features and D will be called target.
2. A,B and C are called independent features. D is called dependent feature.
3. A,B and C are called independent variables. D is called dependent variable.
4. A,B and C is X and D is y. **X = features** and **y = labels**.
5. On X-axis we plot feature and on y-axis we plot label.

## Methods

### Min-Max Scaling

The data will shift to the range of 0 to 1 range.

```Equation
x' = x - min(x) / max(x) - min(x)

x' = scaled value
x = original value

```

### Standard Scaling (Z-Score Normalization)

The range is between -3 to 3. **68%** of data is in the range of **-1 to 1**. **95%** of data is in the range of **-2 to 2**. **99%** of data is in the range of **-3 to 3**.

If your **ML Model** can't handle the negative values then don't use Z-Score normalization.

``` Equation
x' = (x - μ) / σ

or

x' = x - mean(x) / std(x)

```

### Robust Scaling

The range is between -1 to 1.

```Equation

x' = x - median(x) / IQR(x)

```

### Logrithmic Scaling

```Equation

log(x)
        or 
log10(x)

```
