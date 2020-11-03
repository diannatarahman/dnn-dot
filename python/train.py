from keras.callbacks import ModelCheckpoint, EarlyStopping
from builder import *
from keras.optimizers import Adam
import helper


lr = 0.0002
beta_1 = 0.5


# key = 'training_16_old'
# key = 'training'
key = 'training_16'
# key_test = 'calibrated2_16'
model, _ = primary_net(key)
# model = load_model('model62dlog_cloud_13_17_16.h5', custom_objects=custom_objects)
# model.set_weights(model1.get_weights())


# def _add_weight_traverse(model, model1, flag, i):
#     l = len(model.weights)
#     for j in range(l):
#         if j == i:
#             continue
#     ???
#
#
# def load_weights(model, model1):
#     l = len(model.weights)
#     flag = np.zeros(l, dtype=np.bool)
#     for i in range(l):
#         if flag[i]:
#             continue
#     ???



# model = primary_net()
loss = 'mse'
metrics = ['mae', 'mse', 'mape', 'msle', 'logcosh', 'cosine']
model.compile(loss=loss, loss_weights=[0, 0, 1, 1, 1e4, 1], optimizer=Adam(lr=lr, beta_1=beta_1), metrics=metrics)
model.summary()
# model = load_model('model28log.h5', custom_objects=custom_objects)
# model_name = input('Enter model name: ')
model_name = 'model_final_16x16'
print('model name: ' + model_name)
checkpoint = ModelCheckpoint(model_name + '_{epoch:d}.h5', period=1)
checkpoint.set_model(model)
checkpoint_weight = ModelCheckpoint(model_name + '_weights_{epoch:d}.h5', period=1, save_weights_only=True)
checkpoint_weight.set_model(model)
# early_stop = EarlyStopping(monitor='val_1_loss', min_delta=0.1, patience=30, restore_best_weights=True, verbose=1)
callbacks = [TimeHistory(), checkpoint, checkpoint_weight]
# callbacks = [TimeHistory()]
# callbacks = [TimeHistory(), EarlyStopping(monitor='loss', min_delta=0.1, patience=100,
#                                           restore_best_weights=True, verbose=1)]
# early_stop = callbacks.EarlyStopping(patience=200, verbose=1)
# with open('model_hist_data28log.json', 'r') as f:
#     hist_data = json.load(f)
dataset = datasets[key]
# data_size = dataset.data_size
# val_size = dataset.val_size
# data_size = dataset.train_size
# val_size = int(data_size * 0.2)
# choice = np.zeros(data_size, dtype=np.bool)
# choice[np.random.choice(data_size, val_size, replace=False)] = 1
# choice[hist_data['choice']] = 1
choice = datasets[key].data_label
# data_label = datasets[key].data_label
# data_label[data_label == 1] = 2
# data_label[data_label == 0] = choice
# sel = d[key].flatten() == 80
# train_sel = (choice == 0) & sel
# test_sel = (choice == 1) & sel
train_sel = (choice == 0)
val_sel = (choice == 1)
# train_sel = (data_label == 0)
# test_sel = (data_label == 1)
x0 = [PHI_meas[key][train_sel], freq[key][train_sel], d[key][train_sel]]
y0 = [MU[key][train_sel][:, [0]], MU[key][train_sel][:, [1]], MU_norm[key][train_sel][:, [0]],
      MU_norm[key][train_sel][:, [1]], MUa[key][train_sel], MUsp[key][train_sel]]
x1 = [PHI_meas[key][val_sel], freq[key][val_sel], d[key][val_sel]]
y1 = [MU[key][val_sel][:, [0]], MU[key][val_sel][:, [1]], MU_norm[key][val_sel][:, [0]],
      MU_norm[key][val_sel][:, [1]], MUa[key][val_sel], MUsp[key][val_sel]]
# x2 = [PHI_meas[key_test], freq[key_test], d[key_test]]
# MU_norm_iter = MU_iter[key_test] / np.reshape(MU0[key_test], MU0[key_test].shape + (1, 1))
# y2 = [MU_iter[key_test][:, [0]], MU_iter[key_test][:, [1]], MU_norm_iter[:, [0]],
#       MU_norm_iter[:, [1]], MUa[key_test], MUsp[key_test]]
# x0[0] = np.concatenate([x0[0], PHI_meas_seq_pair_masked[key][train_sel]], axis=0)
# for i in range(1, len(x0)):
#     x0[i] = np.concatenate([x0[i], x0[i]], axis=0)
# for i in range(len(y0)):
#     y0[i] = np.concatenate([y0[i], y0[i]], axis=0)
# x1[0] = np.concatenate([x1[0], PHI_meas_seq_pair_masked[key][test_sel]], axis=0)
# for i in range(1, len(x1)):
#     x1[i] = np.concatenate([x1[i], x1[i]], axis=0)
# for i in range(len(y1)):
#     y1[i] = np.concatenate([y1[i], y1[i]], axis=0)
# n = np.where(choice[-N_count[-1]:] == 0)[0].size
# w = 1.5
# sw1 = np.ones(train_size)
# sw2 = np.concatenate([np.ones(train_size-n) * (train_size - n * w) / (train_size - n), np.ones(n) * w])
# sample_weight = [sw1, sw1, sw2, sw2, sw1, sw1]
history = model.fit(x0, y0, validation_data=(x1, y1), batch_size=32, epochs=200, verbose=2, callbacks=callbacks)
# loss_index_batch = (0, 1, 2, 3, 4, 5, 6)
# format_string_batch = ''
# for i, j in enumerate(loss_index_batch):
#     format_string_batch += model.metrics_names[j] + ': %g'
#     if i < len(loss_index_batch)-1:
#         format_string_batch += ', '
# loss_index_epoch = loss_index_batch
# format_string_epoch = format_string_batch
# history = fit_model(model, x0, y0, validation_data=[(x1, y1), (x2, y2)], batch_size=32, epochs=200, verbose=1,
#                     callbacks=callbacks, format_string_batch=format_string_batch, loss_index_batch=loss_index_batch,
#                     format_string_epoch=format_string_epoch, loss_index_epoch=loss_index_epoch, apply_noise=False)
hist_data = {
    'train_start': callbacks[0].train_time_start,
    'train_time': callbacks[0].train_time,
    'epoch_time': callbacks[0].times,
    'history': history.history,
    'choice': np.where(choice == 1)[0].tolist()
}

# Summarize for accuracy training history
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
