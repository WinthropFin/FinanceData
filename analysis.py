#%%
import pandas as pd
import os
# %%
data = pd.read_csv('QRSVX returns.csv')
data.columns = ["date", "QRSVX", "M-index", "R-index","Mkt-Rf","SMB","HML","Rf"]
# %%
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
# %%
x= data["Mkt-Rf"]
y = data["QRSVX"]
window = 12  # months
model = PandasRollingOLS(y=y, x=x, window=window)

print(model.beta.head())
# %%
x= data[["Mkt-Rf","SMB","HML"]]
y = data["QRSVX"]
window = 12  # months
model = PandasRollingOLS(y=y, x=x, window=window)

print(model.beta.head()) 
x= data[["Mkt-Rf","SMB","HML"]]
y = data["QRSVX"] - data["Rf"]
window = 12  # months
model = PandasRollingOLS(y=y, x=x, window=window)

print(model.beta.head()) 
# %%
beta = pd.DataFrame(data = model.beta)
# %%
beta["date"] = pd.date_range('2011-03-01', periods = len(beta), freq= 'm')
beta.plot(x = 'date',y='Mkt-Rf')

# %%
