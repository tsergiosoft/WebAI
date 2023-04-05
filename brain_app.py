from tensorflow import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
import os
import threading


filename1 = 'upl_file1.jpg'
filename2 = 'upl_file2.jpg'
filename3 = 'upl_file3.jpg'


class brain_AI:
    def __init__(self):
        self.res1 = 'no results'
        self.res2 = ''
        self.res3 = ''
        self.learn_active = 'no learning'
        # save_dir = "model/"
        current_file_directory = os.path.dirname(__file__)
        save_dir = os.path.join(current_file_directory, 'model/')

        self.model_path = os.path.normpath(os.path.join(save_dir, 'model1.h5'))
        self.model2_path = os.path.normpath(os.path.join(save_dir, 'model2.h5'))
        self.model3_path = os.path.normpath(os.path.join(save_dir, 'model3.h5'))

        # self.model_path = os.path.join(save_dir, 'model1.h5')
        # self.model2_path = os.path.join(save_dir, 'model2.h5')
        # self.model3_path = os.path.join(save_dir, 'model3.h5')
        self.learn_thread = threading.Thread()

    def analyze_uploaded_images(self):
        # load the model and create predictions on the test set
        model1 = keras.models.load_model(self.model_path)
        model2 = keras.models.load_model(self.model2_path)
        model3 = keras.models.load_model(self.model3_path)

        images = np.zeros(shape=(3, 28, 28))
        # Load image and convert to grayscale
        for i in range(3):
            current_file_directory = os.path.dirname(__file__)
            f = os.path.normpath(os.path.join(current_file_directory, 'static/'+'upl_file' + str(i + 1) + '.jpg'))
            # f = os.path.join('static', 'upl_file' + str(i + 1) + '.jpg')
            img = Image.open(f).convert('L').resize((28, 28))
            # Invert the image
            # img = Image.invert(img)

            img_test = np.asarray(img).astype('float32') / 255
            images[i] = img_test

        predict = []
        predict.append(np.around(model1.predict(images), decimals=2))

        images.reshape(3, 28, 28, 1)
        predict.append(np.around(model1.predict(images), decimals=2))
        predict.append(np.around(model2.predict(images), decimals=2))
        predict.append(np.around(model3.predict(images), decimals=2))

        self.res1 = predict[0]
        self.res2 = predict[1]
        self.res3 = predict[2]

    def start_learning(self):
        if self.learn_thread.is_alive():
            print('already learning!!!')
        else:
            print('learn started')
            self.learn_thread = threading.Thread(target=self.learn)
            self.learn_thread.start()

    def learn(self):
        self.learn_active = 'Learning in process...'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # https://nextjournal.com/gkoehler/digit-recognition-with-keras
        # https://github.com/brianspiering/keras-intro/blob/master/keras-intro.ipynb

        # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # normalizing the data to help with the training
        X_train = train_images / 255.0
        X_test = test_images / 255.0

        # Define the neural network model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1)
        ])
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Create the CNN model
        model2 = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dense(1)
        ])
        model2.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # Create the CNN model
        model3 = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',
                                input_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu',
                                input_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(1)
        ])
        model3.compile(optimizer='adam', loss='mse', metrics=['mae'])


        # training the model and save
        history = model.fit(X_train, train_labels, batch_size=128, epochs=20,verbose=2,validation_data=(X_test, test_labels))
        model.save(self.model_path)

        #Change shape for CNN model
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        # history = model2.fit(X_train, train_labels, batch_size=128, epochs=20, verbose=2,
        #                     validation_data=(X_test, test_labels))
        # model2.save(self.model2_path)

        # history = model3.fit(X_train, train_labels, batch_size=128, epochs=20, verbose=1,
        #                     validation_data=(X_test, test_labels))
        # model3.save(self.model3_path)

        self.learn_active = 'Learning finished'
