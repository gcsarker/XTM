import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, confusion_matrix

from scripts.Utility_Functions import utility_functions

class FPDM:
    
    def __init__(self, data, fpdm_model, lookback = 48, threshold = 0.4):
        self.data = data
        self.fpdm_model = fpdm_model
        self.lookback = lookback
        self.threshold = threshold
        pass
    
    def predict(self, model, batch):
        pass
    
    def barplot_for_specific_fdi(self, real_data, predicted_data, atk_vector_indx):
        inv_scaled_real_data = self.data.inv_scale(real_data)
        injected_data = self.data.inject_fixed_attackvec(inv_scaled_real_data, atk_vector_indx)
        injected_data = self.data.scale(injected_data)
        #injeted_data  = injected_data[self.lookback:]
        print('injected_data shape == {}'.format(injected_data.shape))
        errors = abs(real_data - predicted_data)
        errors_fdi = abs(injected_data - predicted_data)
        t_err = [norm(i) for i in errors]
        t_err_fdi = [norm(i) for i in errors_fdi]
        utility_functions.show_barplot(data_list = [t_err, t_err_fdi], label = ['real','anomaly'], n_bins= 50)
                
    def get_forecasting_errors(self, real_data, predicted_data):
        predictions = self.data.inv_scale(predicted_data)
        real = self.data.inv_scale(real_data)
        mae = utility_functions.MAE(real, predictions)
        mse = utility_functions.MSE(real,predictions)
        rmse = utility_functions.RMSE(real,predictions)
        print('mae : {}, mse : {}, rmse : {}'.format(mae,mse,rmse))
        
    
    def is_fdi(self, real_data, predicted_data, threshold):
        errors = abs(real_data - predicted_data)
        fdi = []
        for i in errors:
            if norm(i)> threshold:
                  fdi.append(1)
            else:
                  fdi.append(0)
        return np.array(fdi)
            
    
    def get_prf(self, real_data, predicted_data, threshold):
        
        fdi = self.is_fdi(real_data, predicted_data, threshold)
        actual = np.zeros((real_data.shape[0],)) #calculating confusion matrix with only real data
        cm0 = confusion_matrix(actual,fdi, labels = [0,1])

        inv_scaled_real_data = self.data.inv_scale(real_data)
        injected_data, _ = self.data.inject_random_attackvec(inv_scaled_real_data)
        injected_data = self.data.scale(injected_data)
        
        fdi = self.is_fdi(injected_data, predicted_data, threshold)
        actual = np.ones((real_data.shape[0],))
        cm1 = confusion_matrix(actual,fdi, labels = [0,1])
        
        cm= cm0 + cm1 #adding confusion matrix with Fdi data
        return np.array(utility_functions.cm2prf(cm.T)), cm.T
    
    
        