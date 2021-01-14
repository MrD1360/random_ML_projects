#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import tensorflow_addons as tfa

#vocab_len=23
#batch_size=16
#buck_t1=8 #max len input (threshold of the first bucket)


# In[2]:


class encoder_model(tf.keras.Model):
    def __init__(self,lat_dim=256,vocab_len=14):#,**kwargs):
        super().__init__()#(**kwargs)
        #encoder
        self.encoder_embedding_layer= tf.keras.layers.Embedding(input_dim=(vocab_len+3),
                                                             output_dim=lat_dim,
                                                             #mask_zero=True,
                                                             name='encoder_embedding')
        self.encoder= tf.keras.layers.LSTM(lat_dim*2,return_sequences=True, return_state=True,name="encoder")#,dropout=0.4)
        


# In[3]:


class decoder_model(tf.keras.Model):
    def __init__(self,lat_dim=256,vocab_len=14,batch_size=16,buck_t1=8):#,**kwargs):
        super().__init__()#(**kwargs)
                #decoder
        self.decoder_embedding_layer= tf.keras.layers.Embedding(input_dim=(vocab_len+3),
                                                             output_dim=(lat_dim),
                                                             #mask_zero=True,
                                                             #embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
                                                             name='decoder_embedding')
        
        #define LuongAttention
        self.attention_mechanism= tfa.seq2seq.LuongAttention(units=(lat_dim*2),
                                                            memory=None,
                                                            memory_sequence_length=batch_size*[buck_t1]
                                                            )
        
        self.decoder= tf.keras.layers.LSTMCell(lat_dim*2, name='decoder_cell')#,dropout=0.4)
        self.attention_decoder= tfa.seq2seq.AttentionWrapper(self.decoder,
                                                            self.attention_mechanism,
                                                            attention_layer_size=(lat_dim*2)
                                                            ) 
        
        
        #dense output layer
        self.dense_layer=tf.keras.layers.Dense(vocab_len+3,  
                          #activation='softmax',
                          name='dense_layer')
        
        #sampler for training                                                 
        training_sampler= tfa.seq2seq.sampler.TrainingSampler() #ScheduledEmbeddingTrainingSampler 
        
        #define training decoder
        self.training_decoder= tfa.seq2seq.BasicDecoder(cell=self.attention_decoder,
                                                       sampler=training_sampler,
                                                       output_layer=self.dense_layer
                                                      )
        

        
    def get_initial_states(self,batch_size,enc_state_h,enc_state_c):
        init_states=self.attention_decoder.get_initial_state(dtype=tf.float32,
                                                            batch_size=batch_size).clone(cell_state=[enc_state_h,enc_state_c])
        return init_states

