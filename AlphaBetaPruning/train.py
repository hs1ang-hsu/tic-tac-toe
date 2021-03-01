from data_generator import set_ll_board_to_array as set_board
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import time
from numpy.random import seed


old = time.time()

with open('data.json', newline='') as jsonfile:
    data = json.load(jsonfile)

x = []
y = []

for key in data.keys():
	x.append(np.array(set_board(int(key))))
	temp = [0] * 25
	for i in data[key]:
		temp[i] = 1
	y.append(np.array(temp))

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=777)

# epoch_list = [10*i for i in range(8,16)]
epoch_list = [160,170,180]
Dense_list = [16,32,48,64]
wrong_avg_low = 10
wrong_avg_list = []
diff_0_list = []
diff_1_list = []
diff_2_list = []


for epoch in epoch_list:
	for dense1 in Dense_list:
		for dense2 in Dense_list:
			for dense3 in Dense_list:
				for dense4 in Dense_list:
					model = Sequential()
					model.add(Dense(dense1, activation='relu', input_dim=(x_train.shape[1])))
					model.add(Dense(dense2, activation='relu'))
					model.add(Dense(dense3, activation='relu'))
					model.add(Dense(dense4, activation='relu'))
					model.add(Dense(y_train.shape[1], activation='relu'))
					model.compile(loss='mse', optimizer="adam")
					# monitor = EarlyStopping(monitor="val_loss", 
						# min_delta=1e-15, patience=15,restore_best_weights=True)
					model.fit(x_train,y_train, validation_data = (x_test,y_test), epochs=epoch)

					diff_2 = 0
					diff_1 = 0
					diff_0 = 0

					for i in range(x.shape[0]):
						temp = x[i].reshape(-1,26)
						pred = model.predict(temp)
						pos = [0,0]
						max_1 = 0
						max_2 = 0
						act = []
						for j in range(pred.shape[1]):
							if temp[0][j] != 0:
								continue
							else:
								if pred[0][j] > max_1:
									max_1 = pred[0][j]
									pos[0] = j+1
								elif pred[0][j] > max_2:
									max_2 = pred[0][j]
									pos[1] = j+1

						for j in range(len(y[i])):
							if y[i][j] == 1:
								act.append(j+1)

						diff = len(set(act)-set(pos))
						if diff == 2:
							diff_2 += 1
						elif diff == 1:
							diff_1 += 1
						elif diff == 0:
							diff_0 += 1

					diff_0_list.append(diff_0)
					diff_1_list.append(diff_1)
					diff_2_list.append(diff_2)
					wrong_avg = (diff_1 + 2*diff_2)/x.shape[0]
					wrong_avg_list.append((epoch,dense1,dense2,dense3,dense4,wrong_avg))

					print(f"wrong_avg: {wrong_avg}")
					if wrong_avg < wrong_avg_low:
						wrong_avg_low = wrong_avg
						epoch_best = epoch
						dense_best = (dense1,dense2,dense3,dense4)
						model_best = model


new = time.time()
print(f"lowest: {wrong_avg_low}")
print(f"epoch: {epoch_best}")
print(f"dense: {dense_best}")
print(f"It cost {new-old} sec")
model_best.save("current_best_model.h5")
