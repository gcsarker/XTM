import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FPDM_Models:
    
    def __init__(self, input_shape,   #input_shape = (lookback,54)
                 lookback = 48,
                 algorithm = 'xtm',
                 head_size = 9,
                 num_heads = 6,
                 ff_dim = 128,
                 num_transformer_blocks = 1,
                 mlp_units=[128],
                 mlp_dropout = 0.1,
                 dropout=0.1,
                 training = False,
                 loss_function = 'mse',
                 optimizer = keras.optimizers.Adam(learning_rate=1e-4),
                 checkpoint_path = "checkpoint\\",
                ):
        
        self.training = training
        self.input_shape = input_shape
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lookback = lookback
        print('Loading '+ self.algorithm+ 'model for FDI presence detection module ...')
        if self.training:
            
                if self.algorithm == 'xtm':
                    self.model = self.build_xtm_model(self.input_shape,
                                          self.head_size,
                                          self.num_heads,
                                          self.ff_dim,
                                          self.num_transformer_blocks,
                                          self.mlp_units,
                                          self.mlp_dropout,
                                          self.dropout
                                         )
                elif self.algorithm == 'cnn_transformer':
                    self.model = self.build_cnn_transformer_model(self.input_shape,
                                          self.head_size,
                                          self.num_heads,
                                          self.ff_dim,
                                          self.num_transformer_blocks,
                                          self.mlp_units,
                                          self.mlp_dropout,
                                          self.dropout
                                         )
                
                elif self.algorithm == 'cnn':
                    self.model = self.build_cnn_model(self.input_shape,
                                                  self.mlp_units,
                                                  self.mlp_dropout,
                                                  self.dropout
                                                 )
                elif self.algorithm == 'cnn_lstm':
                    self.model = self.build_cnn_lstm_model(self.input_shape,
                                                       self.mlp_units,
                                                       self.mlp_dropout,
                                                       self.dropout
                                                      )
                
                elif self.algorithm == 'transformer':
                    self.model = self.build_transformer_model(self.input_shape,
                                          self.head_size,
                                          self.num_heads,
                                          self.ff_dim,
                                          self.num_transformer_blocks,
                                          self.mlp_units,
                                          self.mlp_dropout,
                                          self.dropout
                                         )
                
                else:
                    self.model = self.build_xtm_model(self.input_shape,
                                          self.head_size,
                                          self.num_heads,
                                          self.ff_dim,
                                          self.num_transformer_blocks,
                                          self.mlp_units,
                                          self.mlp_dropout,
                                          self.dropout
                                        )
                
                self.model = self.compile_model(self.model,self.input_shape,self.loss_function, self.optimizer)
                
                self.cpt_path = self.checkpoint_path+self.algorithm+"\\"
                #self.cpt_path = os.path.join(os.getcwd(), "checkpoint\\"+algorithm+"\\")
                #if not os.path.exists(self.cpt_path):
                #    os.mkdir(self.cpt_path)
                
                self.callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
                                  keras.callbacks.ModelCheckpoint(filepath = self.cpt_path, save_weights_only=True, monitor= 'val_loss',mode = 'min', save_best_only=True)]
        else:
            #self.checkpoint_file_path = os.path.join(os.getcwd(), "checkpoint\\"+"fpdm_"+self.algorithm+".h5")
            self.checkpoint_file_path = self.checkpoint_path+"fpdm_"+self.algorithm+".h5"
            print("loading "+self.algorithm+" from saved models...")
            self.model = keras.models.load_model(self.checkpoint_file_path)
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Dense(ff_dim, activation = 'relu')(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(inputs.shape[-1], activation = 'relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x+res

        # Feed Forward Part
        #x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        #x = layers.Dropout(dropout)(x)
        #x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        #x = layers.LayerNormalization(epsilon=1e-6)(x)
        #return x + res
    
        
    def build_xtm_model(self,input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,mlp_dropout,dropout):
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        #x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        x = layers.LSTM(128, activation = 'tanh',return_sequences=True)(x)
        x = layers.LSTM(128, activation = 'tanh')(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = [layers.Dense(1)(x) for i in range(54)]
        return keras.Model(inputs, outputs)
    
    
    def build_cnn_transformer_model(self,input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout,mlp_dropout):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.Conv1D(128, 9, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        #x = layers.Conv1D(128, 9, activation='relu')(x)
    
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        #outputs = [layers.Dense(1)(x) for i in range(input_shape[1])]
        outputs = [layers.Dense(1)(x) for i in range(54)]
        return keras.Model(inputs, outputs)

    
    def build_cnn_model(self,input_shape, mlp_units, mlp_dropout, dropout):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.Conv1D(128, 9, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 9, activation='relu')(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = [layers.Dense(1)(x) for i in range(54)]
        return keras.Model(inputs, outputs)
    
    
    def build_cnn_lstm_model(self,input_shape, mlp_units, mlp_dropout, dropout):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.Conv1D(128, 9, activation='relu')(x)
        x = layers.MaxPooling1D(3)(x)
        x = layers.LSTM(128, activation = 'tanh',return_sequences=True)(x)
        x = layers.LSTM(128, activation = 'tanh')(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = [layers.Dense(1)(x) for i in range(54)]
        return keras.Model(inputs, outputs)
    
    
    def build_transformer_model(self,input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,mlp_dropout,dropout):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = [layers.Dense(1)(x) for i in range(54)]
        return keras.Model(inputs, outputs)
    
    
    def compile_model(self, model, input_shape, loss_function, optimizer):
        losses = [loss_function for i in range(input_shape[-1])]
        model.compile(loss = losses, optimizer = optimizer)
        return model
    
    def train(self, train_gen, val_gen, steps_per_epoch, validation_steps, epochs = 50, save_model = False):
        training_history= self.model.fit(train_gen,
                           steps_per_epoch = steps_per_epoch,
                           validation_data = val_gen,
                           validation_steps = validation_steps,
                           epochs=epochs,
                           callbacks=self.callbacks
                          )
        if save_model:
            print('saving model ...')
            self.model.save(self.checkpoint_file_path)
            history_df = pd.DataFrame(training_history.history)
            #history_df.to_excel(os.path.join(os.getcwd(), 'checkpoint\\training_history\\'+self.algorithm+'_training_history.xlsx'))
            print('saving training history...')
            history_df.to_excel(self.checkpoint_path+ "training_history\\"+self.algorithm+"_training_history.xlsx")
            
    
    def get_model_summary(self):
        return self.model.summary()
    
    def plot_model(self):
        return keras.utils.plot_model(self.model)
    
    
    def predict(self):
        pass
    
    def is_fdi(self):
        pass
    