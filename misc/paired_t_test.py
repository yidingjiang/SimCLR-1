import pandas as pd 
from scipy import stats

df = pd.read_csv("results_jac.csv")
print(stats.ttest_rel(df['simclr'], df['simclr_jacobian']))