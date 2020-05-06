import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data.csv')
pd.pivot_table(df, index = 'Name', values = ['Overall', 'Potential']).sort_values('Overall', ascending = False)
df_overalll_analysis = df[['Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties','Overall', 'Potential']]

h_labels = [x.replace('_', ' ').title() for x in 
            list(df_overalll_analysis.select_dtypes(include=['number', 'bool']).columns.values)]
corr = df_overalll_analysis.corr()
fig, ax = plt.subplots(figsize=(10,6))
sns_plot = sns.heatmap(corr, annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
sns_plot.figure.savefig('heatmap.png', dpi = 200)

df_composure = pd.pivot_table(df, index = 'Composure', values = ['Overall', 'Potential']).sort_values('Composure', ascending = False)
df_composure = df_composure.reset_index()

#finding NAN's and replacing with 0 
df_overalll_analysis.isnull().sum(axis = 0)
df_overalll_analysis = df_overalll_analysis.fillna(0)


#Dependent var = overall and potential
columns = ['Overall', 'Potential']
X = df_overalll_analysis.drop(columns, axis =1)
y = df_overalll_analysis.iloc[:, 22].values
y1 = df_overalll_analysis.iloc[:, 23].values
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

#Linear Regression
from sklearn.linear_model import LinearRegression, Lasso
lr = LinearRegression()
lr.fit(X_train, y_train)
np.mean(cross_val_score(lr, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
y_pred_lr = lr.predict(X_test)
mean_absolute_error(y_test, y_pred_lr)

alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i)/100)
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3)))

plt.plot(alpha, error)

#SVR
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
np.mean(cross_val_score(svr, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
y_pred_svr = svr.predict(X_test)
mean_absolute_error(y_test, y_pred_svr)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
parameters_rf = {
        'n_estimators' : [5, 50, 250],
        'max_depth' : [2,4,8,16,32, None]
        }
cv_rf = GridSearchCV(rf, parameters_rf, cv = 5)
cv_rf.fit(X_train, y_train)
cv_rf = cv_rf.best_estimator_

#np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
y_pred_rf = cv_rf.predict(X_test)
mean_absolute_error(y_test, y_pred_rf)

#MLP regressor
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor()
parameters_nn = {
        'hidden_layer_sizes' : [(10,), (50,), (100,)],
        'activation' : ['relu', 'tanh', 'logistic'],
        'learning_rate' : ['constant', 'invscaling', 'adaptive']
        }
cv_nn = GridSearchCV(nn, parameters_nn, cv = 5)
cv_nn.fit(X_train, y_train)
#np.mean(cross_val_score(nn, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
cv_nn = cv_nn.best_estimator_
y_pred_nn = cv_nn.predict(X_test)
mean_absolute_error(y_test, y_pred_nn)
#Going with MLPRegressor withot hyper parameter tuning though other objects are trained
#and can be used if needed
nn.fit(X_train, y_train)
y_pred_nn1 = nn.predict(X_test)
mean_absolute_error(y_test, y_pred_nn1)

#XG Boost Regressor
import xgboost as xgb
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train, y_train)
np.mean(cross_val_score(xg_reg, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
y_pred_xg = xg_reg.predict(X_test)
mean_absolute_error(y_test, y_pred_xg)






















