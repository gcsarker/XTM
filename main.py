import os
from sklearn.preprocessing import MinMaxScaler

from scripts.Data import Data
from scripts.FPDM import FPDM
from scripts.FPDM_Models import FPDM_Models
from scripts.LDM_Data import LDM_Data
from scripts.LDM_Model import LDM_Model
from scripts.Utility_Functions import utility_functions

data_path = os.path.join(os.getcwd(), "dataset\\hourly_dataset.csv")
atk_vector_path = os.path.join(os.getcwd(), "dataset\\fdi_attack_vector.xlsx")
mm_scaler = MinMaxScaler((0,1))
lookback = 48 # see pash 48 datapoints
delay = 1     # forecast future 1 datapoint
batch_size=32 # batch size for training FPDM model
steps=1        # step

n_sensor_measurement  = 54 # number of sensor_measurements
fpdm_input_shape = (lookback,n_sensor_measurement) # input shape for fdi presence detection module
fpdm_training = False # To specify whether to train fpdm
algorithm = 'xtm' # Choose which algorithm to use for fpdm
checkpoint_path = "checkpoint\\"

ldm_training = False # To specify whether to train ldm
load_data_ = False # To specify whether to load prepared randomly injected data for ldm training from memory

data = Data(data_path = data_path, atk_vector_path = atk_vector_path, scaler = mm_scaler)
fpdm_model = FPDM_Models(input_shape = fpdm_input_shape, algorithm = algorithm, training = fpdm_training)

if fpdm_training:
    steps_per_epoch = (data.len_train_set - lookback)//batch_size
    validation_steps = (data.len_val_set - lookback)//batch_size
    fpdm_model.train(data.train_gen, data.val_gen, steps_per_epoch, validation_steps, epochs = 50)

ldm_data = LDM_Data(data, fpdm_model.model, algorithm = algorithm, training = ldm_training, load_data_ = load_data_)
    
# fDI presence detection module
fpdm = FPDM(data, fpdm_model, 48, threshold = 0.4)

print('\nplotting histogram showing how fpdm can detect attack vector 5 from our attack vector dataset')
fpdm.barplot_for_specific_fdi(real_data = ldm_data.test_set, predicted_data = ldm_data.test_predictions, atk_vector_indx = 5)

print('\ncalculating MAE, MSE and RMSE score of FPDM forecasting model on test data')
fpdm.get_forecasting_errors(ldm_data.test_set,ldm_data.test_predictions)

prf, cm = fpdm.get_prf(ldm_data.test_set, ldm_data.test_predictions, threshold = 0.4)
print('\nCalculating precision recall f1-score of FPDM on test data')
print('precision = {}, recall = {}, f1-score = {}'.format(prf[0], prf[1], prf[2]))

#ldm_data = LDM_Data(data, fpdm_model.model, algorithm = algorithm, training = ldm_training)
ldm_model = LDM_Model(ldm_data = ldm_data, algorithm = algorithm, training= ldm_training)

if ldm_training:
    ldm_data.generate_randomly_injected_fdi_data()
    ldm_model.train()
else:
    #for injecting only test set
    injected_test_data, fdi_location_test = ldm_data.inject_fdi(ldm_data.test_set)

# To get the ROC curve of location detection on test data
print("\nThe roc curve for location detection module")
ldm_model.get_roc_curve(injected_test_data, ldm_data.test_predictions, fdi_location_test)

# location prediction and Calculating precision recall f1-score on test data
print('\ncalculating precision recall f1 score of LDM')
location_prediction = ldm_model.predict(real_data = injected_test_data, predicted_data = ldm_data.test_predictions)
ldm_model.get_ldm_prf(real_location_data = fdi_location_test, predicted_location_data = location_prediction)

