import os
import numpy as np
from tqdm import tqdm

class LDM_Data:
    
    def __init__(self, data, model, algorithm, lookback = 48, delay = 1, batch_size= 50, training = False, load_data_ = True):
        
        self.data = data
        self.model = model
        self.algorithm = algorithm
        self.lookback = lookback
        self.delay = delay
        self.batch_size = batch_size
        self.training = training
        self.load_data_ = load_data_
        self.data_path = os.path.join(os.getcwd(),"checkpoint\\")
        
        self.train_gen = self.model2_generator(self.data.train_set, 
                     lookback=self.lookback, 
                     delay=self.delay, 
                     min_index=0, 
                     max_index=None, 
                     shuffle=False, 
                     step=1,
                     batch_size = self.batch_size)
        
        self.val_gen = self.model2_generator(self.data.val_set,
                     lookback=self.lookback, 
                     delay=self.delay, 
                     min_index=0, 
                     max_index=None, 
                     shuffle=False, 
                     step=1,
                     batch_size = self.batch_size)
        
        self.test_gen = self.model2_generator(self.data.test_set,
                     lookback=self.lookback, 
                     delay=self.delay, 
                     min_index=0, 
                     max_index=None, 
                     shuffle=False, 
                     step=1,
                     batch_size = self.batch_size)
        if self.training:
            self.train_set, self.train_predictions = self.prepare_dataset_for_ldm(self.data.train_set, self.train_gen, filename = 'LDM_'+self.algorithm+'_train')
            self.val_set, self.val_predictions = self.prepare_dataset_for_ldm(self.data.val_set, self.val_gen, filename = 'LDM_'+self.algorithm+'_val')
        
        self.test_set, self.test_predictions = self.prepare_dataset_for_ldm(self.data.test_set, self.test_gen, filename = 'LDM_'+self.algorithm+'_test')
    
    def inject_fdi(self, data):
        # injecting random attack vectors to data 
        inv_scaled_data = self.data.inv_scale(data)
        injected_data, fdi_location = self.data.inject_random_attackvec(inv_scaled_data)
        injected_data = self.data.scale(injected_data)
        return injected_data, fdi_location
    
    def generate_randomly_injected_fdi_data(self):
        # injecting random attack vectors to train, val, test for training LDM. This function needs to run manually
        print('\ninjecting datasets with random attack vector...')
        print("\nPreparing train, validation test set for ldm training...")
        self.train_set, self.fdi_location_train = self.inject_fdi(self.train_set)
        self.val_set, self.fdi_location_val = self.inject_fdi(self.val_set)
        self.test_set, self.fdi_location_test = self.inject_fdi(self.test_set)
    
    def randomize_ldm_data(self):
        pass
    
    def save_data(self,data, filename):
        np.savetxt(os.path.join(self.data_path, filename),data)
    
    def load_data(self, filename):
        return np.loadtxt(os.path.join(self.data_path, filename))
    
    def model2_generator(self, data, lookback, delay, min_index, max_index, shuffle=False, batch_size=50, step=1):
        if max_index is None:
            max_index = len(data)- delay #8
        i = min_index + lookback
        while 1:
            if shuffle:
                  rows = np.random.randint(min_index + lookback, max_index+1, size=batch_size)
            else:
                if i + batch_size > max_index: #+ batch_size 
                    rows = np.arange(i,max_index+1)
                    i = min_index + lookback
                else:
                    rows = np.arange(i, min(i + batch_size, max_index+1))
                    i += len(rows)
            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows),data.shape[-1]))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j]+ delay-1][:] # + delay-1
            yield samples, np.array([k for k in targets.T]) #[i for i in targets.T]
            
    def prepare_dataset_for_ldm(self, dataset, gen, filename):
        if self.load_data_:
            print("\nLoading data...")
            real = self.load_data(filename+'.txt')
            prediction = self.load_data(filename+'_predictions.txt')
            return real, prediction
        
        real = np.zeros((len(dataset)-self.lookback, dataset.shape[-1]))
        prediction = np.zeros((len(dataset)-self.lookback, dataset.shape[-1]))
        
        print("\nPreparing test set for LDM (real and Predicted from FPDM) ...")
        for i in tqdm(range(0,len(dataset)-self.lookback, self.batch_size)):
            test_data = next(gen)
            pred = np.array(self.model.predict_on_batch(test_data[0]))
            pred = np.reshape(pred,(pred.shape[0],pred.shape[1])).T
            rows = np.arange(i,min(len(dataset)-self.lookback,i+self.batch_size))
            for j, row in enumerate(rows):
                real[row] = test_data[1].T[j]
                prediction[row] = pred[j]
        
        print("\nSaving Data...")
        self.save_data(real, filename+'.txt')
        self.save_data(prediction, filename+'_predictions.txt')
        return real, prediction