import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models # type: ignore
from math import sqrt, ceil
import numpy as np




#====================    DIGIT RECOGNITION    ====================#



tf.random.set_seed(1234)
np.random.seed(1234)
fig = plt.figure(figsize=(9,6))



#---------------------  Prepare Data  --------------------#



(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input shape
dimen = x_train.shape[1] * x_train.shape[2]


# flatten the 3D array to a 2D array
X_train = x_train.reshape((x_train.shape[0], dimen)).astype('float32')
X_test  = x_test.reshape((x_test.shape[0], dimen)).astype('float32')


# normalize the data [0,255] -> [-1,1]
X_train = (X_train - 128) / 128
X_test = (X_test - 128) / 128



# display train images 
def show_train_images(dig):
    dim = ceil(sqrt(dig))
    for i in range(1,dig+1):
        plt.subplot(dim,dim,i)
        plt.imshow(x_train[i],cmap='gray')
        plt.axis('off')
        plt.title(y_train[i])
    plt.tight_layout()
    plt.show()


# show_train_images(34)



#--------------------  Create Network  -------------------#



model = models.Sequential()

model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(units=128, activation='relu', name='layer1'))
model.add(layers.Dense(units=64, activation='relu', name='layer2'))
model.add(layers.Dense(units=32, activation='relu', name='layer3'))
model.add(layers.Dense(units=10, activation='softmax', name='layer4'))

model.summary()

layer1, layer2, layer3, layer4 = model.layers

W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
W4,b4 = layer4.get_weights()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    X_train,y_train,epochs=40,batch_size=100
)

evaluations = model.evaluate(X_test,y_test)
print('loss and accuracy: ', evaluations)


# display test images 
def show_test_images(dig):
    dim = ceil(sqrt(dig))
    for i in range(1,dig+1):
        predictions = model.predict(X_test[i].reshape(1,dimen))
        # print('predictions: ', predictions)
        plt.subplot(dim,dim,i)
        plt.imshow(x_test[i],cmap='gray')
        plt.axis('off')
        plt.title(y_test[i])
    plt.tight_layout()
    plt.show()

show_test_images(25)