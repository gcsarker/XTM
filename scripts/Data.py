import numpy as np
import pandas as pd

class Data:
    
    def __init__(self, data_path, atk_vector_path, scaler, lookback = 48, delay = 1, batch_size=32, step=1):
        self.data_path = data_path
        self.atk_vector_path = atk_vector_path
        self.mm = scaler
        self.lookback = lookback
        self.delay = delay
        self.step = step
        self.batch_size = batch_size
        
        self.training_set, self.columns = Data.load_benign_data(self.data_path)
        self.atk_vectors, self.label_set = Data.load_atk_vectors(self.atk_vector_path)
        
        self.mm = self.mm.fit(self.training_set)
        self.scaled = self.scale(self.training_set)
        
        self.train_set, self.val_set, self.test_set = self.train_val_test_split()
        self.len_train_set, self.len_val_set, self.len_test_set = len(self.train_set), len(self.val_set), len(self.test_set)
        
        self.train_gen = self.generator(self.train_set,
                                    lookback=self.lookback,
                                    delay=self.delay,
                                    min_index=0,
                                    max_index=None, 
                                    shuffle=True, 
                                    step=self.step,
                                    batch_size = self.batch_size)

        self.val_gen = self.generator(self.val_set, 
                                  lookback=self.lookback, 
                                  delay=self.delay, 
                                  min_index=0, 
                                  max_index=None, 
                                  shuffle=True, 
                                  step=self.step,
                                  batch_size = self.batch_size)
    
    @staticmethod
    def load_benign_data(data_path):
        print("\nLoading Benign Hourly Data...")
        df = pd.read_csv(data_path)
        columns = df.columns[1:]
        
        print("\nFeatures Selected {}".format(columns))
        training_set = df[columns].astype(float).to_numpy()
        
        return training_set, columns
    
    
    @staticmethod
    def load_atk_vectors(atk_vector_path):
        print("\nLoading False Data Injection Attack Vectors...")
        atk_vectors = pd.read_excel(atk_vector_path)
        atk_vectors.drop("Unnamed: 0", axis=1, inplace = True)
        atk_vectors = atk_vectors.to_numpy()
        label_set = []
        for i in atk_vectors:
          label_set.append([0 if k==0 else 1 for k in i])
        label_set = np.array(label_set)
        
        print("\nshape of Attack vector", atk_vectors.shape)
        print("shape of labels ", label_set.shape)
        
        return atk_vectors, label_set
    
    
    def scale(self, data):
        return self.mm.transform(data)
    
    def inv_scale(self, scaled_data):
        return self.mm.inverse_transform(scaled_data)
        
    def train_val_test_split(self, test_set_percentage= 0.2, val_set_percentage = 0.5):
        test_split = int(self.scaled.shape[0]*test_set_percentage)
        test_set = self.scaled[-test_split:]
        train_set = self.scaled[:-test_split]
        
        val_split = int(test_set.shape[0]*val_set_percentage)
        final_test_set = test_set[-val_split:]
        val_set = test_set[:-val_split]

        print("\nTraining set shape: ",train_set.shape)
        print("Validation set shape ",val_set.shape)
        print("Test Set Shape ",final_test_set.shape)
        
        return train_set, val_set, final_test_set
    
    def inject_fixed_attackvec(self, data, atk_vector_indx, scale = True):
        
        atk_vector = self.atk_vectors[atk_vector_indx]
        for dt_point in data:
          for i in range(len(atk_vector)):
            if atk_vector[i] !=0:
              dt_point[i] = dt_point[i]+atk_vector[i]
        return data
    
    def inject_random_attackvec(self, data):
        location_set = []
        for row,sample in enumerate(data):
            indx = np.squeeze(np.random.choice(self.atk_vectors.shape[0],size = 1, replace = True))
            atk_vector = self.atk_vectors[indx,]
            data[row] = sample+atk_vector
            location_set.append(self.label_set[indx,])
        return data, np.array(location_set)
    
    def generator(self, data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=1):
        if max_index is None:
            max_index = len(data)- delay #8
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(min_index + lookback, max_index+1, size=batch_size)
            else:
                if i + batch_size-1 > max_index: #+ batch_size 
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index+1))
                i += 1
            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows),data.shape[-1]))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j]+ delay-1][:] # + delay-1
            yield samples, [i for i in targets.T]
    