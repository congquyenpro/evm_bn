import pickle
import earnedvaluemanagement as evm


inputName = "ModelSettings_Example_2.xlsx"

# Tải model
with open('case2.pkl', 'rb') as f:
    case = pickle.load(f)

# Tải sample của model
with open('case_sample2.pkl', 'rb') as f:
    case_sample = pickle.load(f)

evm.excel_posterior(case_sample,inputName)