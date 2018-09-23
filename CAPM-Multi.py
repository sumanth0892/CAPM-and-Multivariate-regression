#Capital Asset Pricing Model
#Ri = Rf + Bi * (Rmld - Rf)
#For a security i, its returns is defined as Ri and its beta as Bi
#Linear regression with Scipy
from scipy import stats
stock_returns = [0.065,0.0265,-0.0678,-0.001,0.0346]
mkt_returns = [0.055,-0.09,-0.041,0.045,0.022]
beta,alpha,r_value,p_value,std_err = stats.linregress(stock_returns,mkt_returns)
print(beta)
print(alpha)

#SML is described as
#E(Ri) = Rf + Bi*(E(Rm) - Rf)
#Terms of market risk premium and risk-free rate and return on asset i
#An ordinart least squared regression
import numpy as np
import statsmodels.api as sm
#Generate sample data
num_periods = 9
all_values  =np.array([np.random.random(8) for i in range(num_periods)])

#Filter the data
y_values = all_values[:,0] #First column values as Y
x_values = all_values[:,1:] #All otehr values as X

x_values = sm.add_constant(x_values)
results = sm.OLS(y_values,x_values).fit() #Regress and fit the model
