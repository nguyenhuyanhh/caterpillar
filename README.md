# Caterpillar Tube Pricing

Dis mashine learning, it kool

## Instructions

Just run `$ python predict_linear.py` or `$ python predict_xgb.py`. For the latter, make sure [XGBoost](https://xgboost.readthedocs.io/en/latest/build.html) is installed.

Results are in `out.csv`.

## Version 1: Linear Regression

### Model

For bracket tubes:

* base tube cost = `f(diameter, wall, length, num_bends, bend_radius)` - linear regression from cost of 1-bracket tubes
* cost coefficients - linear regression of `cost*quantity` over `quantity`

For non-bracket tubes, cost coefficients are omitted.

### Results

| Model | Results (rmsle)
| --- | ---
| `cost` as power function of `quantity` | 0.721308
| `cost*quantity` as linear function of `quantity` | 0.680392

## Version 2: XGBoost

### Model

* Log transform for `cost`
* One-hot encoding for `supplier`, `bracket_pricing`, tube end forms
* `annual_usage`, `min_order_quantity`, quantity of components

### Results

| Model | Results (rmsle)
| --- | ---
| with log transform for `cost` | 0.487616
| with `annual_usage` and `min_order_quantity` | 0.387799
| with one-hot encoding of `supplier` | 0.320696
| with quantity of components | 0.263805