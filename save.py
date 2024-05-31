import earnedvaluemanagement as evm
import pickle

inputName = "ModelSettings_Example_1.xlsx"

PRJ = evm.project_reader(inputName)
case = evm.build_model(PRJ)
case_sample = evm.sample_model(case)

with open('case.pkl', 'wb') as f:
    pickle.dump(case, f)

# Lưu sample của model
with open('case_sample.pkl', 'wb') as f:
    pickle.dump(case_sample, f)