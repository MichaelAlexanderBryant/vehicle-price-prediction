#####import libraries and data

#import libraries
import pandas as pd

#supress warnings
import warnings
warnings.filterwarnings("ignore")

#import data
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')


#import ML package
from sklearn.svm import SVR

#support vector regressor final model
svr_model = SVR(C=24, coef0= 0.9, gamma='scale', kernel='poly')
svr_model.fit(X_train,y_train)
tpred_svr=svr_model.predict(X_test)


# #####pickle
# import pickle

# outfile = open('support_vector_regression_model.pkl', 'wb')
# pickle.dump(svr_model,outfile)
# outfile.close()
