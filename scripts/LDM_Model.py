import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scripts.Utility_Functions import utility_functions

class LDM_Model:
    
    def __init__(self, ldm_data,
                 input_shape = (54,),
                 output_heads = 54,
                 algorithm = 'xtm',
                 epochs = 100,
                 batch_size = 32,
                 loss_function= 'binary_crossentropy',
                 optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                 training= False):
        self.ldm_data = ldm_data
        self.input_shape = input_shape
        self.output_heads = output_heads
        self.algorithm = algorithm
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.training = training
        self.model = self.ldm_model()
        if self.training:
            self.compile_model()
            self.callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] 
                        
        else:
            self.checkpoint_file_path = os.path.join(os.getcwd(), "checkpoint\\"+"ldm_"+self.algorithm+".h5")
            print("loading ldm model trained with predictions from "+self.algorithm+" from the saved models...")
            self.model = keras.models.load_model(self.checkpoint_file_path)

            
    
    def ldm_model(self):
        pred_inp = keras.Input(shape=self.input_shape)
        fdi_inp = keras.Input(shape = self.input_shape)
        concat = layers.concatenate([pred_inp,fdi_inp])
        dense1 = layers.Dense(128, activation = 'relu')(concat)
        dense2 = layers.Dense(128,activation = 'relu')(dense1)
        dense3 = layers.Dense(128, activation = 'relu')(dense2)
        outputs = [layers.Dense(1, activation = 'sigmoid')(dense3) for i in range(self.output_heads)]
        
        return keras.Model([pred_inp,fdi_inp], outputs)

    def compile_model(self):
        losses = [self.loss_function for i in range(self.input_shape[-1])]
        self.model.compile(loss = losses, optimizer = self.optimizer)
    
    
    def train(self, save_model = False):
        training_history= self.model.fit([self.ldm_data.train_predictions,self.ldm_data.train_set],
                                         [i for i in self.ldm_data.fdi_location_train.T],
                                         validation_data = ([self.ldm_data.val_predictions,self.ldm_data.val_set],[k for k in self.ldm_data.fdi_location_val.T]),
                                         epochs=self.epochs,
                                         batch_size = self.batch_size,
                                         callbacks=self.callbacks
                                        )
        
        if save_model:
            self.model.save(self.checkpoint_file_path)
            history_df = pd.DataFrame(training_history.history)
            history_df.to_excel(os.path.join(os.getcwd(), 'checkpoint\\training_history\\'+self.algorithm+'_location_training_history.xlsx'))

        
    def get_model_summary(self):
        return self.model.summary()
    
    def plot_model(self):
        return keras.utils.plot_model(self.model)
    
    def predict(self, real_data, predicted_data, threshold = 0.5):
        raw_location_prediction = np.squeeze(self.model.predict([predicted_data,real_data]))
        raw_location_prediction = raw_location_prediction.T
        location_prediction = []
        for i in raw_location_prediction:
            temp = []
            for k in i:
                if k> threshold:
                    temp.append(1)
                elif (threshold == 1.0) and (k == 1.0):
                    temp.append(0)
                elif (threshold == 0) and (k == 0):
                    temp.append(1)
                else:
                    temp.append(0)  
            location_prediction.append(temp)
        return np.array(location_prediction)
    
    def roc_curve(self, real_data , predicted_data, threshold, location_set):
        cms = []
        tprs0 = []
        fprs0 = []
        tprs1 = []
        fprs1 = []
        loc = self.predict(real_data, predicted_data, threshold)
        for i in range(54):
            cms.append(confusion_matrix(location_set.T[i],loc.T[i], labels = [0,1]).T)

        for i in cms:
            #class 0
            tp = i[0][0]
            fp = i[0][1]
            fn = i[1][0]
            tn = i[1][1]

            tpr0 = tp/(tp+fn)
            fpr0 = fp/(fp+tn)

            #class 1
            tp = i[1][1]
            fp = i[1][0]
            fn = i[0][1]
            tn = i[0][0]

            tpr1 = tp/(tp+fn)
            fpr1 = fp/(fp+tn)
  
            tprs0.append(tpr0)
            fprs0.append(fpr0)
            tprs1.append(tpr1)
            fprs1.append(fpr1)
        tprs0,fprs0,tprs1,fprs1 = np.array(tprs0),np.array(fprs0),np.array(tprs1),np.array(fprs1)
        return np.mean(tprs0), np.mean(fprs0), np.mean(tprs1), np.mean(fprs1)
    
    def get_roc_curve(self, real_data , predicted_data, location_set):
        thresholds = []
        tpr_class0 = []
        fpr_class0 = []
        tpr_class1 = []
        fpr_class1 = []

        ts = np.linspace(0,1.0, num = 11)

        for i in ts:
            temp = self.roc_curve(real_data = real_data, predicted_data = predicted_data ,threshold = i, location_set = location_set)
            thresholds.append(i)
            tpr_class0.append(temp[0])
            fpr_class0.append(temp[1])
            tpr_class1.append(temp[2])
            fpr_class1.append(temp[3])
         
        utility_functions.plot_roc_curve([tpr_class0,tpr_class1],[fpr_class0,fpr_class1], n_classes=2)
    
    def get_ldm_prf(self, real_location_data, predicted_location_data):
        prf_loc = []
        for i in range(54):
            cm = confusion_matrix(real_location_data.T[i],predicted_location_data.T[i], labels = [0,1])
            prf_loc.append(np.array(utility_functions.cm2prf(cm)))
        
        prf_loc = np.array(prf_loc)
        prf_loc = np.mean(prf_loc, axis = 0)
        print('precision = {}, recall = {}, f1 score = {}'.format(prf_loc[0],prf_loc[1],prf_loc[2]))
        return prf_loc