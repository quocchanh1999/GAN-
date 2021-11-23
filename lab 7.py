# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:20:30 2021

@author: chanh
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import BatchNormalization,Flatten,Input,Dense,LeakyReLU,Dropout,Activation,Reshape

batch_size = 128 # batch size để đưa vào mô hình huấn luyện
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_digits=np.concatenate([x_train,x_test])
all_digits=(all_digits.astype("float32")-127.5)/127.5 #đưa tấm hình về [-1,1]
all_digits=np.reshape(all_digits,(-1,28,28,1)) #reshape toàn bộ dữ liệu lại
dataset=tf.data.Dataset.from_tensor_slices(all_digits)  #Đưa dữ liệu về dạng tensor
dataset=dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)    #shuffle dữ lieu

#mô hình của mạng Discriminator
discriminator=keras.Sequential()

discriminator.add(Input(shape=(28,28,1)))

discriminator.add(BatchNormalization())
discriminator.add(Flatten())

discriminator.add(Dense(units=1024))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(rate=0.3))

discriminator.add(Dense(units=512))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(rate=0.3))

discriminator.add(Dense(units=256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(rate=0.3))

discriminator.add(Dense(units=1))

discriminator.add(Activation(activation='sigmoid',name="sigmoid"))

print(discriminator.summary())

#mô hình của mạng Generator
latent_dim=100
generator=keras.Sequential()

generator.add(Input(shape=(latent_dim,)))

generator.add(Dense(units=256))
generator.add(LeakyReLU(alpha=0.2))

generator.add(Dense(units=512))
generator.add(LeakyReLU(alpha=0.2))

generator.add(Dense(units=1024))
generator.add(LeakyReLU(alpha=0.2))

generator.add(Dense(units=28*28))
generator.add(Activation(activation='tanh',name='tanh')) #Đưa về một tấm ảnh nên dùng tanh để về [-1,1]

generator.add(Reshape((28,28,1)))
generator.add(BatchNormalization())

print(generator.summary())

class Gan(keras.Model):
    def __init__(self,discriminator,generator,latent_dim):
        super(Gan,self).__init__()
        self.discriminator=discriminator
        self.generator=generator
        self.latent_dim=latent_dim
    def compile(self,d_optimizer,g_optimizer,loss_function):
        super(Gan,self).compile()
        self.d_optimizer=d_optimizer
        self.g_optimizer=g_optimizer
        self.loss_function=loss_function
    def train_step(self,real_images):
        # tạo batch size từ dữ liệu đầu vào
        batch_size=tf.shape(real_images)[0]
        random_latent_vectors=tf.random.normal(shape=(batch_size,self.latent_dim))
        
        # Tạo ra ảnh giả từ mạng generator
        generated_images=self.generator(random_latent_vectors)
        
        # nối ảnh giả và ảnh thật làm bộ dữ liệu cho mô hình discriminator
        combined_images=tf.concat([generated_images,real_images],axis=0)
        
        # Ở đây đánh nhãn ảnh thật là 0, ảnh giả là 1
        labels=tf.concat(
            [tf.ones((batch_size,1)),tf.zeros((batch_size,1))],axis=0
            )
        # trick để add thêm noise cho nhãn
        labels+=0.05*tf.random.uniform(tf.shape(labels))
        
        # huấn luyện discriminator
        with tf.GradientTape() as tape:
            predictions=self.discriminator(combined_images) #Dự đoán 
            d_loss=self.loss_function(labels,predictions) #Đưa kết quả dự đoán và gro
        grads=tape.gradient(d_loss,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads,self.discriminator.trainable_weights)
            ) #cập nhật weights cho mô hình discrimunator
        
        # lặp lại bước tạo noise như ở trên
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # tạo mộ bộ nhãn sao cho đều là 1 để đánh lừa discriminator
        misleading_labels = tf.zeros((batch_size, 1))

        # Train generator (lưu ý k update weights cho discriminator)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_function(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

#overwrite callback để lúc huấn luyện xem ảnh in ra
from IPython.display import Image,display
class GANMonitor(keras.callbacks.Callback):
    def __init__(self,num_img=3,latent_dim=100):
        self.num_img=num_img
        self.latent_dim=latent_dim
        
    def on_epoch_end(self,epoch,logs=None):
        random_latent_vectors=tf.random.normal(shape=(self.num_img,self.latent_dim))
        generated_images=self.model.generated_images(random_latent_vectors) ##tạo ảnh
        generated_images=generated_images*127.5+127.5 #Đưa ảnh về [0,255]
        generated_images.numpy()
        print("epoch",epoch)
        for i in range(self.num_img):
            img=keras.preprocessing.image.array_to_img(generated_images[i])
            display(img) #in ảnh
            
epochs=30
latent_dim=100
gan=Gan(discriminator=discriminator,generator=generator,latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_2=0.5),
    loss_function=keras.losses.BinaryCrossentropy(),
    )            
gan.fit(
        dataset,epochs=epochs,callbacks=[GANMonitor(num_img=3,latent_dim=latent_dim)]
                
        )




