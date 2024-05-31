""" import pandas as pd
import numpy as np
import pymc3 as pm

# Hàm tính phân vị
def trace_quantiles(x):
    quantiles = np.percentile(x, [5, 50, 95])
    return pd.DataFrame(quantiles.reshape(-1, 1), index=[5, 50, 95], columns=['quantiles'])

# Hàm tính độ lệch chuẩn
def trace_sd(x):
    return pd.Series(np.std(x, 0), name='sd')

# Hàm tính trung bình
def trace_mean(x):
    return pd.Series(np.mean(x), name='mean')

# Tạo một trace giả định từ một từ điển dữ liệu
trace_data = {
    'param1': np.random.normal(loc=0, scale=1, size=1000),
    'param2': np.random.normal(loc=2, scale=0.5, size=1000)
}

# Tạo mô hình PyMC3 và trace từ dữ liệu giả định
with pm.Model() as model:
    param1 = pm.Normal('param1', mu=0, sigma=1)
    param2 = pm.Normal('param2', mu=2, sigma=0.5)
    trace = pm.sample_prior_predictive(samples=1000)

# Tạo DataFrame từ trace
trace_df = pd.DataFrame(trace)

# Tạo tệp Excel chứa trace
outputName = "trace_output.xlsx"
trace_df.to_excel(outputName, sheet_name="Trace")
 """
import pandas as pd
import numpy as np
import pymc3 as pm

# Function to calculate quantiles
import arviz as az

def trace_quantiles(x):
  quantiles = az.quantiles(x, [5, 50, 95])
  return pd.DataFrame(quantiles)

# Function to calculate standard deviation
def trace_sd(x):
    return pd.Series(np.std(x, 0), name='sd')

# Function to calculate mean
def trace_mean(x):
    return pd.Series(np.mean(x), name='mean')

# Assume trace data
trace_data = {
    'param1': np.random.normal(loc=0, scale=1, size=1000),
    'param2': np.random.normal(loc=2, scale=0.5, size=1000)
}

# Create trace from dummy data
with pm.Model() as model:
    param1 = pm.Normal('param1', mu=0, sigma=1)
    param2 = pm.Normal('param2', mu=2, sigma=0.5)
    trace = pm.sample_prior_predictive(samples=1000)

# Calculate summary using pm.summary
all_names = ['param1', 'param2']
summary = pm.summary(trace, var_names=all_names, stat_funcs={'mean': trace_mean, 'sd': trace_sd, 'quantiles': trace_quantiles})

# Write the result to an Excel file
output_name = 'summary.xlsx'
summary.to_excel(output_name, sheet_name='Summary')


