from helper import *
from keras.models import Model
from keras.layers import *
from functools import partial, update_wrapper
from custom_functions import multiply_constant, multiply_constant_reciprocal
from keras import regularizers, initializers


def fn_mask(x):
    from scipy.io import loadmat
    mask = loadmat('mask.mat')['mask'].astype(np.float)
    return multiply_constant(x, mask)


def fn_freq(x):
    from helper import const_freq
    return multiply_constant_reciprocal(x, const_freq, const_freq)


def fn_d(x):
    from helper import const_d
    return multiply_constant_reciprocal(x, const_d)**2


# reg_l1 = 1e-5
# reg_l2 = 1e-6
# rate_1 = 0.1
# rate_2 = 0.4
# rate_3 = 0.4
activation = 'softplus'
LOCATIONS = len(Xb)
# mask = loadmat('mask.mat')['mask']
# fn = partial(multiply_mask, mask=mask)

# class AccumulatorCell(Layer):
#
#     def __init__(self, units, channels=1, activation=None, **kwargs):
#         self.units = units
#         self.channels = channels
#         self.activation = activations.get(activation)
#         self.state_size = [self.units * self.units for _ in range(self.channels + 1)]
#         super(AccumulatorCell, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         input_dim = input_shape[-1] - 2 + 2*self.channels
#         # self.embeddings = self.add_weight(shape=(self.units * self.units, self.channels, input_dim),
#         #                                   initializer='uniform', name='embeddings')
#         # self.embeddings_bias = self.add_weight(shape=(self.channels,), initializer='zeros', name='embeddings_bias')
#         self.embeddings = self.add_weight(shape=(self.units, self.channels), initializer='uniform', name='embeddings')
#         self.kernel1 = self.add_weight(shape=(input_dim, self.channels), initializer='glorot_uniform', name='kernel1')
#         self.bias1 = self.add_weight(shape=(self.channels,), initializer='zeros', name='bias1')
#         # self.kernel2 = self.add_weight(shape=(self.units, self.channels), initializer='glorot_uniform',
#         #                                name='kernel2')
#         # self.bias2 = self.add_weight(shape=(self.channels,), initializer='zeros', name='bias2')
#         self.built = True
#
#     def call(self, inputs, states):
#         prev_outputs = K.concatenate([K.expand_dims(s, axis=1) for s in states[:-1]], axis=1)
#         counter = states[-1]
#         l = int(inputs.shape[1])
#         loc = K.slice(inputs, (0, l-2), (-1, 2))
#         if K.dtype(loc) != 'int32':
#             loc = K.cast(loc, 'int32')
#         loc0 = K.slice(loc, (0, 0), (-1, 1))
#         loc1 = K.slice(loc, (0, 1), (-1, 1))
#         signal = K.slice(inputs, (0, 0), (-1, l-2))
#         # h = K.dot(signal, self.kernel1)
#         # h = K.bias_add(h, self.bias1, data_format='channels_last')
#         # if self.activation is not None:
#         #     h = self.activation(h)
#         # h = K.expand_dims(h, 1)
#         # index = K.squeeze((loc0-1)*self.units+(loc1-1), -1)
#         # tmp_kernel = K.gather(self.embeddings, index)
#         # h = K.sum(h * tmp_kernel, axis=-1)
#         # h = K.bias_add(h, self.embeddings_bias, data_format='channels_last')
#         # if self.activation is not None:
#         #     h = self.activation(h)
#         # h = K.dot(h, self.kernel2)
#         # h = K.bias_add(h, self.bias2, data_format='channels_last')
#         # if self.activation is not None:
#         #     h = self.activation(h)
#         embed_loc = K.gather(self.embeddings, loc)
#         embed_loc = K.reshape(embed_loc, (-1, np.prod(embed_loc.shape[1:])))
#         h = K.dot(K.concatenate([signal, embed_loc]), self.kernel1)
#         h = K.bias_add(h, self.bias1, data_format='channels_last')
#         if self.activation is not None:
#             h = self.activation(h)
#         # h = K.dot(h, self.kernel2)
#         # h = K.bias_add(h, self.bias2, data_format='channels_last')
#         # if self.activation is not None:
#         #     h = self.activation(h)
#         h2 = K.cast(K.all(K.greater(loc, 0), axis=1, keepdims=True), dtype=K.floatx())
#         one_hot0 = K.one_hot(loc0-1, self.units)
#         one_hot0 = K.permute_dimensions(one_hot0, (0, 2, 1))
#         one_hot1 = K.one_hot(loc1-1, self.units)
#         one_hot = one_hot0 * one_hot1
#         one_hot = K.reshape(one_hot, (-1, np.prod(one_hot.shape[1:])))
#         counter += one_hot * h2
#         outputs = K.expand_dims(one_hot, 1) * K.expand_dims(h * h2) + prev_outputs
#         next_states = []
#         for i in range(self.channels):
#             o = K.slice(outputs, (0, i, 0), (-1, 1, -1))
#             o = K.squeeze(o, 1)
#             if i == 0:
#                 output = o
#             next_states.append(o)
#         next_states.append(counter)
#         return output, next_states


def rnn_block(inputs, channels=1):
    outputs = []
    for _in in inputs:
        states = RNN(AccumulatorCell(LOCATIONS, channels), return_state=True)(_in)
        output = []
        counter = states[channels + 1]
        # counter_rbf = states[channels + 2]
        # counter_src = states[channels * 2 + 3]
        # counter_rbf_src = states[channels * 2 + 4]
        # rbf0 = states[channels * 2 + 5]
        # gate = states[channels + 3]
        # counter_rbf0 = states[channels + 3]
        # counter_rbf1 = states[channels + 4]
        # counter_rbf01 = states[channels + 5]
        # flag0 = states[channels + 6]
        # flag1 = states[channels + 7]
        # div = Lambda(lambda x: K.clip(x, 1, None))(counter)
        units = int(np.sqrt(int(states[1].shape[-1])))
        # _counter = Reshape((units, 1, units))(counter)
        for i, s in enumerate(states[1:(channels+1)]):
            # if i > 0:
            #     s = Lambda(lambda x: x[0] / x[1])([s, div])
            # _o = Reshape((units, 1, units))(s)
            # _o = concatenate([_o, _counter], axis=2)
            # _o = TimeDistributed(Conv1D(16, 3, padding='same', data_format='channels_first'))(_o)
            # _o = TimeDistributed(Conv1D(1, 3, activation='relu', padding='same', data_format='channels_first'))(_o)
            # _o = TimeDistributed(Flatten())(_o)
            # _o = TimeDistributed(Dense(units))(_o)
            # _o1 = Activation('sigmoid')(_o)
            # _o = Lambda(lambda x: x[0] * (K.max(x[1], axis=-1, keepdims=True) -
            #                               K.min(x[1], axis=-1, keepdims=True)))([_o1, _o])
            # _o = Reshape((1, units*units))(_o)
            # output.append(_o)
            s = Lambda(lambda x: x[0] / x[1])([s, counter])
            # s = Lambda(lambda x: (x[0] + (x[1] * x[2] / K.clip(x[3], 1, None))) / K.clip(x[4], 1, None))(
            #     [s1, rbf0, s2, counter_src, counter])
            o = Reshape((1, units*units))(s)
            output.append(o)
        # counter = Reshape((1, units*units))(counter)
        # output.append(counter)
        if len(output) > 1:
            output = concatenate(output, axis=1)
        else:
            output = output[0]
        # output = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, 1, -1)))(output)
        # transpose = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(output)
        # output = concatenate([output, transpose], axis=1)
        # output = Dropout(0.1)(output)
        # output1 = Conv1D(16, 3, activation='relu', padding='same', data_format='channels_first')(output)
        # output2 = Conv1D(16, 5, activation='relu', padding='same', data_format='channels_first')(output)
        # output3 = Conv1D(16, 7, activation='relu', padding='same', data_format='channels_first')(output)
        # output4 = Conv1D(16, units, activation='relu', padding='same', data_format='channels_first')(output)
        # output = concatenate([output1, output2, output3, output4], axis=1)
        # output = Conv1D(16, units, activation='relu', padding='same', data_format='channels_first')(output)
        # output = BatchNormalization(axis=1)(output)
        # output = LeakyReLU(lrelu_alpha)(output)
        # output = ELU()(output)
        outputs.append(output)
    return outputs


# def contrast_filter_block(inputs, units=None):
def contrast_filter_block(inputs, input_mask=None, units=None, n_out=1):
    # if units is None:
    #     units = [512]
    # gru_combine = 2
    # n_out = int(inputs[0].shape[1])
    # n_out = 1
    # outputs = []
    # for _in in inputs:
    #     #filtering
    #     output = Conv1D(16, 3, padding='same', data_format='channels_first')(_in)
    #     output = Conv1D(16, 3, padding='same', data_format='channels_first')(output)
    #     # output = ActivityRegularization(l1=reg_l1, l2=reg_l2)(output)
    #     output = BatchNormalization(axis=1)(output)
    #     output = ELU()(output)
    #
    #     #skip connection
    #     _output = Conv1D(16, 3, padding='same', data_format='channels_first')(output)
    #     _output = BatchNormalization(axis=1)(_output)
    #     _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #     _output = Conv1D(16, 3, padding='same', data_format='channels_first')(_output)
    #     _output = BatchNormalization(axis=1)(_output)
    #     _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #     _output = Conv1D(16, 3, padding='same', data_format='channels_first')(_output)
    #     _output = BatchNormalization(axis=1)(_output)
    #     output = add([output, _output])
    #     output = LeakyReLU(lrelu_alpha)(output)
    #     # output = SpatialDropout1D(rate_1)(output)
    #
    #     output = Conv1D(64, 3, strides=3, data_format='channels_first')(output)
    #     output = Conv1D(64, 5, strides=5, data_format='channels_first')(output)
    #     output = BatchNormalization(axis=1)(output)
    #     output = ReLU()(output)
    #
    #     output = Conv1D(16, 1, data_format='channels_first')(output)
    #     output = BatchNormalization(axis=1)(output)
    #     output = ReLU()(output)
    #     output = Flatten()(output)
    #     outputs.append(output)
    # if len(outputs) > 1:
    #     outputs = concatenate(outputs)
    # else:
    #     outputs = outputs[0]
    #
    # l = K.int_shape(outputs)[-1]
    # layer = Dense(l, activation='relu')
    # outputs = layer(outputs)
    # bg_filters = outputs

    #feature extraction
    # outputs = Bidirectional(GRU(256))(inputs[0], mask=input_mask)
    # outputs = Bidirectional(GRU(16, return_sequences=True))(inputs[0], mask=input_mask)
    # outputs = Bidirectional(CuDNNGRU(256))(outputs)

    # if PREPROCESSING_MODE == 'SEQUENTIAL':
    #     freq_param = inputs[-2]
    #     d_param = inputs[-1]
    #     outputs = []
    #     bg_filters = []
    #     for _in in inputs[:-2]:
    #         output = _in
    #         # nonzero = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(output)
    #
    #         bg_filter = Conv2D(16, 1, activation='relu')(output)
    #         bg_filter = Lambda(lambda x: K.mean(x, axis=-1))(bg_filter)
    #         bg_filter = Conv1D(16, 1, activation='relu', data_format='channels_first')(bg_filter)
    #         bg_filter = GlobalAveragePooling1D()(bg_filter)
    #         bg_filters.append(bg_filter)
    #
    #         # nonzero = Lambda(lambda x: K.cast(K.not_equal(x, 0), dtype=K.floatx()) * 9)(nonzero)
    #         # nonzero = AveragePooling2D(pool_size=3, strides=1, padding='same')(nonzero)
    #         # div = Lambda(lambda x: K.clip(x, 1, None))(nonzero)
    #         output = Conv2D(64, 3, padding='same')(output)
    #         # output = Lambda(lambda x: x[0] / x[1])([output, div])
    #         output = BatchNormalization(axis=1)(output)
    #         output = LeakyReLU(lrelu_alpha)(output)
    #
    #         units = K.int_shape(output)[-1]
    #         _output = output
    #         for i in range(5):
    #             if int(_output.shape[2]) <= 16:
    #                 break
    #             # nonzero = Lambda(lambda x: K.cast(K.not_equal(x, 0), dtype=K.floatx()) * 16)(nonzero)
    #             # nonzero = AveragePooling2D(pool_size=4, strides=2, padding='same')(nonzero)
    #             # div = Lambda(lambda x: K.clip(x, 1, None))(nonzero)
    #             _output = Conv2D(32, 4, strides=2, padding='same')(_output)
    #             # _output = Lambda(lambda x: x[0] / x[1])([_output, div])
    #             _output = BatchNormalization(axis=1)(_output)
    #             _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #         _output = Conv2D(32, 3, padding='same')(_output)
    #         _output = BatchNormalization(axis=1)(_output)
    #         _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #         l = K.int_shape(_output)[1:]
    #         _output = Flatten()(_output)
    #         _output = Dense(128)(_output)
    #         _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #         output_param = Dense(128)(freq_param)
    #         output_param = LeakyReLU(lrelu_alpha)(output_param)
    #         # output_add1 = multiply([_output, output_param])
    #         _output = multiply([_output, output_param])
    #
    #         output_param = Dense(128)(d_param)
    #         output_param = LeakyReLU(lrelu_alpha)(output_param)
    #         # output_add2 = multiply([_output, output_param])
    #         _output = multiply([_output, output_param])
    #
    #         # _output = add([_output, output_add1, output_add2])
    #         _output = Dense(np.prod(l))(_output)
    #         # _output = Dropout(0.1)(_output)
    #         _output = LeakyReLU(lrelu_alpha)(_output)
    #         _output = Reshape(l)(_output)
    #
    #         _output = Conv2D(32, 3, padding='same')(_output)
    #         _output = BatchNormalization(axis=1)(_output)
    #         _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #         for j in range(i):
    #             _output = Conv2DTranspose(32, 4, strides=2, padding='same')(_output)
    #             _output = BatchNormalization(axis=1)(_output)
    #             _output = LeakyReLU(lrelu_alpha)(_output)
    #
    #         _output = Lambda(lambda x: K.slice(x, (0, 0, 0, 0), (-1, -1, units, units)))(_output)
    #         _output = Conv2D(64, 3, padding='same')(_output)
    #         output = add([output, _output])
    #         output = BatchNormalization(axis=1)(output)
    #         output = LeakyReLU(lrelu_alpha)(output)
    #
    #         output = Conv2D(16, 3, padding='same')(output)
    #         l = K.int_shape(output)[-1]
    #         output = Reshape((-1, l))(output)
    #         output = Conv1D(16 * n_out, 3, padding='same', data_format='channels_first')(output)
    #         while int(output.shape[2]) > 16:
    #             output = Conv1D(16 * n_out, 4, strides=2, padding='same', data_format='channels_first')(output)
    #         # output = Conv2D(n_out, 3, padding='same', activity_regularizer=regularizers.l2(1e-6))(output)
    #         output = BatchNormalization(axis=1)(output)
    #         output = ReLU()(output)
    #         output = Flatten()(output)
    #         outputs.append(output)
    #     # outputs = Flatten()(outputs)
    #     # layer = Concatenate()
    #     # outputs = layer([outputs, freq_param, d_param])
    #     if len(outputs) > 1:
    #         outputs = concatenate(outputs)
    #     else:
    #         outputs = outputs[0]
    #
    #     l = K.int_shape(outputs)[-1]
    #     layer = Dense(l, activation='relu')
    #     outputs = layer(outputs)
    #     # bg_filters = outputs
    # elif PREPROCESSING_MODE == 'FIXED':
    freq_param = inputs[-2]
    d_param = inputs[-1]
    outputs = []
    bg_filters = []
    for _in in inputs[:-2]:
        s = K.int_shape(_in)
        l = int(np.sqrt(s[-1]))
        bg_filter = Conv1D(l+1, l, strides=l, activation='relu', data_format='channels_first')(_in)
        bg_filter = Conv1D(l+1, 1, activation='relu', data_format='channels_first')(bg_filter)
        bg_filter = GlobalAveragePooling1D(data_format='channels_first')(bg_filter)
        bg_filters.append(bg_filter)

        output = Conv1D(l+1, 3, padding='same', data_format='channels_first')(_in)
        output = Conv1D(l+1, 3, padding='same', data_format='channels_first')(output)
        output = BatchNormalization(axis=1)(output)
        # output = LeakyReLU(lrelu_alpha)(output)
        output = ELU()(output)

        # l = int(np.sqrt(K.int_shape(output)[-1]))
        # output1 = Conv1D(64, l, strides=l, data_format='channels_first')(output)
        # output1 = BatchNormalization(axis=1)(output1)
        # output1 = LeakyReLU(lrelu_alpha)(output1)
        #
        # output2 = Conv1D(16, l + 1, dilation_rate=l, data_format='channels_first')(output)
        # output2 = BatchNormalization(axis=1)(output2)
        # output2 = LeakyReLU(lrelu_alpha)(output2)
        # output2 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(output2)

        # output = concatenate([output1, output2], axis=1)
        output = Conv1D(4*(l+1), 3, padding='same', data_format='channels_first')(output)
        output = BatchNormalization(axis=1)(output)
        output = LeakyReLU(lrelu_alpha)(output)

        output = Conv1D(4*(l+1), l, strides=l, data_format='channels_first')(output)
        # while True:
        #     if l <= 16:
        #         if len(s) == 4:
        #             output = Reshape((64, -1))(output)
        #         break
        #     if len(s) == 3:
        #         l_ = s[-1] // l
        #         output = Reshape((64, l, l_))(output)
        #     output = Conv2D(64, 4, strides=2, padding='same')(output)
        #     s = K.int_shape(output)
        #     l = s[2]
        # output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(output)
        # output = LocallyConnected1D(64, l, strides=l, use_bias=False)(output)
        # output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(output)
        output = Conv1D(4*(l+1), 3, padding='same', data_format='channels_first')(output)
        # output = concatenate([_output, output], axis=1)
        output = BatchNormalization(axis=1)(output)
        output = LeakyReLU(lrelu_alpha)(output)
        output = Conv1D(4*(l+1), 3, padding='same', data_format='channels_first')(output)
        output = BatchNormalization(axis=1)(output)
        output = LeakyReLU(lrelu_alpha)(output)

        _output = Conv1D(2*(l+1), 3, padding='same', data_format='channels_first')(output)
        _output = BatchNormalization(axis=1)(_output)
        _output = LeakyReLU(lrelu_alpha)(_output)

        sh = K.int_shape(_output)[1:]
        _output = Flatten()(_output)
        _output = Dense(128)(_output)
        _output = LeakyReLU(lrelu_alpha)(_output)

        output_param = Dense(128)(freq_param)
        output_param = LeakyReLU(lrelu_alpha)(output_param)
        # output_add1 = multiply([_output, output_param])
        _output = multiply([_output, output_param])

        output_param = Dense(128)(d_param)
        output_param = LeakyReLU(lrelu_alpha)(output_param)
        # output_add2 = multiply([_output, output_param])
        _output = multiply([_output, output_param])

        # _output = add([_output, output_add1, output_add2])
        _output = Dense(np.prod(sh))(_output)
        # _output = Dropout(0.1)(_output)
        _output = LeakyReLU(lrelu_alpha)(_output)
        _output = Reshape(sh)(_output)

        _output = Conv1D(2*(l+1), 3, padding='same', data_format='channels_first')(_output)
        _output = BatchNormalization(axis=1)(_output)
        _output = LeakyReLU(lrelu_alpha)(_output)

        _output = Conv1D(4*(l+1), 3, padding='same', data_format='channels_first')(_output)
        output = add([output, _output])
        output = BatchNormalization(axis=1)(output)
        output = LeakyReLU(lrelu_alpha)(output)

        output = Conv1D((l+1) * n_out, 3, padding='same', data_format='channels_first')(output)
        while int(output.shape[2]) > 16:
            output = Conv1D((l+1) * n_out, 4, strides=2, padding='same', data_format='channels_first')(output)
        output = BatchNormalization(axis=1)(output)
        output = ReLU()(output)
        output = Flatten()(output)
        outputs.append(output)
        # output = Flatten()(output)
        # layer = Concatenate()
        # output = layer([output, freq_param, d_param])
        # bg_filters = output
    if len(outputs) > 1:
        outputs = concatenate(outputs)
    else:
        outputs = outputs[0]

    l = K.int_shape(outputs)[-1]
    layer = Dense(l, activation='relu')
    outputs = layer(outputs)
        # bg_filters = outputs
    # else:
    #     raise NotImplementedError()

    bg_filters = concatenate(bg_filters + inputs[-2:])
    # outputs = []
    # for i, u in enumerate(units):
    #     o = Masking()(inputs[i])
    #     o = Bidirectional(GRU(u//2, dropout=0.1))(o)
    #     outputs.append(o)
    # outputs = concatenate(outputs)
    # outputs = Dropout(0.1)(outputs)
    # outputs = Dense(512, activation='tanh')(outputs)
    # outputs = Masking()(inputs[0])
    # outputs = Bidirectional(GRU(units//2))(outputs)
    # outputs = Bidirectional(CuDNNGRU(units//2))(inputs[0])
    # outputs = Bidirectional(GRU(units//2))(inputs[0], mask=input_mask)

    # bg_filters = Dense(256, activation='tanh')(outputs)
    # bg_filters = LeakyReLU(lrelu_alpha)(bg_filters)

    # param_mul = Dense(units, activation='sigmoid')(concatenate([freq_param, d_param]))
    # param_mul = LeakyReLU(lrelu_alpha)(param_mul)
    # layer = Multiply()
    # outputs = layer([outputs, param_mul])
    # layer = Concatenate()
    # outputs = concatenate([outputs, freq_param, d_param])
    # outputs = concatenate(outputs + [freq_param, d_param])

    # outputs = concatenate([outputs, freq_param_mul, d_param_mul])
    # outputs = Dense(units)(outputs)

    # layer = Activation('tanh')
    # layer = LeakyReLU(lrelu_alpha)
    # outputs = layer(outputs)
    # outputs = add([outputs, freq_param_mul, d_param_mul])
    # bg_filters = Conv1D(1, 1, data_format='channels_first')(outputs)
    # outputs = Reshape((1, int(outputs.shape[1]), int(outputs.shape[2])))(outputs)
    # outputs = Conv2D(2, 1)(outputs)
    # outputs = BatchNormalization(axis=1)(outputs)
    # layer = LeakyReLU(lrelu_alpha)
    # outputs = layer(outputs)

    # outputs = Conv1D(channels, 1, data_format='channels_first')(outputs)
    # outputs = BatchNormalization(axis=1)(outputs)
    # outputs = LeakyReLU(lrelu_alpha)(outputs)
    #
    # _outputs = Conv1D(16, 3, padding='same', data_format='channels_first')(outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(16, 3, padding='same', data_format='channels_first')(_outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(16, 3, padding='same', data_format='channels_first')(_outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(channels, 1, data_format='channels_first')(_outputs)
    # outputs = add([outputs, _outputs])
    # outputs = BatchNormalization(axis=1)(outputs)
    # outputs = LeakyReLU(lrelu_alpha)(outputs)
    # layer = Flatten()
    # outputs = layer(outputs)

    # bg_filters = Flatten()(bg_filters)
    # bg_filters = concatenate([bg_filters, freq_param, d_param])
    # bg_filters = outputs

    # outputs = Dense(512)(outputs)

    # _outputs = Conv1D(64, 5, padding='same', data_format='channels_first')(outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(64, 5, padding='same', data_format='channels_first')(_outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(64, 5, padding='same', data_format='channels_first')(_outputs)
    # _outputs = BatchNormalization(axis=1)(_outputs)
    # _outputs = LeakyReLU(lrelu_alpha)(_outputs)
    #
    # _outputs = Conv1D(1, 1, data_format='channels_first')(_outputs)
    # outputs = add([outputs, _outputs])
    # outputs = BatchNormalization(axis=1)(outputs)
    # outputs = LeakyReLU(lrelu_alpha)(outputs)
    # outputs = Flatten()(outputs)

    # params = [Dense(16)(x) for x in inputs[1:]]
    # params = [LeakyReLU(lrelu_alpha)(x) for x in params]
    # params = Dense(64)(concatenate(params))
    # params = LeakyReLU(lrelu_alpha)(params)
    # params = Dense(512, activation='tanh')(params)
    # outputs = multiply([outputs, params])
    # outputs = Dropout(0.5)(outputs)
    # output1 = Conv1D(64 * n_out, 15, strides=15, data_format='channels_first')(inputs)
    # output1 = BatchNormalization(axis=1)(output1)
    # output1 = LeakyReLU(lrelu_alpha)(output1)
    #
    # output1 = Conv1D(64 * n_out, 1, data_format='channels_first')(output1)
    # output1 = BatchNormalization(axis=1)(output1)
    # output1 = LeakyReLU(lrelu_alpha)(output1)
    #
    # output1 = Conv1D(64 * n_out, 1, data_format='channels_first')(output1)
    # output1 = BatchNormalization(axis=1)(output1)
    # output1 = LeakyReLU(lrelu_alpha)(output1)
    # output1 = [Lambda(lambda x: K.slice(x, (0, 60 * i, 0), (-1, 60, -1)))(output1) for i in range(n_out)]
    # output1 = [Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outputs) for outputs in output1]
    # output1 = concatenate(output1, axis=1)
    # output1 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(output1)

    # output1 = Conv1D(32 * n_out, 4, strides=2, padding='same', data_format='channels_first')(output1)
    # output1 = BatchNormalization(axis=1)(output1)
    # output1 = LeakyReLU(lrelu_alpha)(output1)

    # output1 = Conv1D(64 * n_out, 4, strides=2, padding='same', data_format='channels_first')(output1)
    # output1 = BatchNormalization(axis=1)(output1)
    # output1 = LeakyReLU(lrelu_alpha)(output1)

    # output2 = Conv1D(64 * n_out, PHI_meas['training'].shape[-1]//30, dilation_rate=15,
    #                  data_format='channels_first')(inputs)
    # output2 = BatchNormalization(axis=1)(output2)
    # output2 = LeakyReLU(lrelu_alpha)(output2)

    # output2 = Conv1D(64 * n_out, 1, data_format='channels_first')(output2)
    # output2 = BatchNormalization(axis=1)(output2)
    # output2 = LeakyReLU(lrelu_alpha)(output2)

    # output2 = Conv1D(64 * n_out, 1, data_format='channels_first')(output2)
    # output2 = BatchNormalization(axis=1)(output2)
    # output2 = LeakyReLU(lrelu_alpha)(output2)

    # outputs = add([output1, output2])
    # outputs = BatchNormalization(axis=1)(outputs)
    # outputs = LeakyReLU(lrelu_alpha)(outputs)
    # outputs = Flatten()(outputs)
    return layer, outputs, bg_filters


# def reshape_fixed_inputs(inputs, n_out=1):
#     l = K.int_shape(inputs[0])[-1] // n_out
#     input_ = Lambda(lambda x: K.expand_dims(x, 1))(inputs[0])
#     fn = partial(multiply_constant, const_freq, const_freq, True)
#     freq_param = Lambda(fn, name='lambda_div_freq')(inputs[-2])
#     fn = partial(multiply_constant, const_d, 0, True)
#     d_param = Lambda(fn, name='lambda_div_d')(inputs[-1])
#     outputs = []
#     for i in range(n_out):
#         o = Lambda(lambda x: K.slice(x, (0, 0, i * l), (-1, -1, l)))(input_)
#         outputs.append(o)
#     if len(outputs) > 1:
#         outputs = concatenate(outputs, axis=1)
#     else:
#         outputs = outputs[0]
#     return outputs, freq_param, d_param


def u_net(inputs, aux_inputs=[], n_out=None, mul=1):
    return_list = True
    if isinstance(inputs, tf.Tensor):
        inputs = [inputs]
        return_list = False
    if not isinstance(n_out, list):
        n_out = [n_out for _ in range(len(inputs))]
    for i, _in in enumerate(inputs):
        if n_out[i] is None:
            n_out[i] = int(_in.shape[1])
    aux_outputs = [[] for _ in range(5)]
    for i, _in in enumerate(aux_inputs):
        aux_outputs[i].append(_in)
    # dense = Dense(np.prod(MU['training'].shape[2:]))(inputs)
    # dense = Reshape((1,) + MU['training'].shape[2:])(dense)

    # dense = Reshape((64, int(MU['training'].shape[2]/8), int(MU['training'].shape[3]/8)))(dense)
    #
    # convolve = Conv2DTranspose(32, 4, strides=2, padding='same')(dense)
    # convolve = BatchNormalization(axis=1)(convolve)
    # convolve = LeakyReLU(lrelu_alpha)(convolve)
    #
    # convolve = Conv2DTranspose(16, 4, strides=2, padding='same')(convolve)
    # convolve = BatchNormalization(axis=1)(convolve)
    # convolve = LeakyReLU(lrelu_alpha)(convolve)
    #
    # convolve = Conv2DTranspose(8, 4, strides=2, padding='same')(convolve)
    # convolve = BatchNormalization(axis=1)(convolve)
    # convolve = LeakyReLU(lrelu_alpha)(convolve)
    #
    # convolve = Conv2DTranspose(4, 1, padding='same')(convolve)
    # convolve = BatchNormalization(axis=1)(convolve)
    # convolve = LeakyReLU(lrelu_alpha)(convolve)
    #
    # convolve = Conv2DTranspose(2, 1, padding='same')(convolve)
    # convolve = BatchNormalization(axis=1)(convolve)
    # convolve = LeakyReLU(lrelu_alpha)(convolve)
    #
    # convolve = Conv2DTranspose(1, 1, padding='same')(convolve)
    # # convolve = LeakyReLU(lrelu_alpha)(convolve)
    # convolve = Lambda(fn)(convolve)

    # skip connection
    # input_ = Input((n_out, int(inputs.shape[2]), int(inputs.shape[3])))
    for i, _in in enumerate(inputs):
        convolve1 = Conv2D(4 * mul, 3, padding='same')(_in)
        convolve1 = BatchNormalization(axis=1)(convolve1)
        convolve1 = LeakyReLU(lrelu_alpha)(convolve1)

        convolve1 = Conv2D(4 * mul, 3, padding='same')(convolve1)
        convolve1 = BatchNormalization(axis=1)(convolve1)
        convolve1 = LeakyReLU(lrelu_alpha)(convolve1)
        aux_outputs[0].append(convolve1)

        convolve2 = Conv2D(8 * mul, 4, strides=2, padding='same')(convolve1)
        convolve2 = BatchNormalization(axis=1)(convolve2)
        convolve2 = LeakyReLU(lrelu_alpha)(convolve2)

        convolve2 = Conv2D(8 * mul, 3, padding='same')(convolve2)
        convolve2 = BatchNormalization(axis=1)(convolve2)
        convolve2 = LeakyReLU(lrelu_alpha)(convolve2)

        convolve2 = Conv2D(8 * mul, 3, padding='same')(convolve2)
        convolve2 = BatchNormalization(axis=1)(convolve2)
        convolve2 = LeakyReLU(lrelu_alpha)(convolve2)
        aux_outputs[1].append(convolve2)

        convolve3 = Conv2D(16 * mul, 4, strides=2, padding='same')(convolve2)
        convolve3 = BatchNormalization(axis=1)(convolve3)
        convolve3 = LeakyReLU(lrelu_alpha)(convolve3)

        convolve3 = Conv2D(16 * mul, 3, padding='same')(convolve3)
        convolve3 = BatchNormalization(axis=1)(convolve3)
        convolve3 = LeakyReLU(lrelu_alpha)(convolve3)

        convolve3 = Conv2D(16 * mul, 3, padding='same')(convolve3)
        convolve3 = BatchNormalization(axis=1)(convolve3)
        convolve3 = LeakyReLU(lrelu_alpha)(convolve3)
        aux_outputs[2].append(convolve3)

        convolve4 = Conv2D(32 * mul, 4, strides=2, padding='same')(convolve3)
        convolve4 = BatchNormalization(axis=1)(convolve4)
        convolve4 = LeakyReLU(lrelu_alpha)(convolve4)

        convolve4 = Conv2D(32 * mul, 3, padding='same')(convolve4)
        convolve4 = BatchNormalization(axis=1)(convolve4)
        convolve4 = LeakyReLU(lrelu_alpha)(convolve4)

        convolve4 = Conv2D(32 * mul, 3, padding='same')(convolve4)
        convolve4 = BatchNormalization(axis=1)(convolve4)
        convolve4 = LeakyReLU(lrelu_alpha)(convolve4)
        aux_outputs[3].append(convolve4)

        convolve5 = Conv2D(64 * mul, 4, strides=2, padding='same')(convolve4)
        convolve5 = BatchNormalization(axis=1)(convolve5)
        convolve5 = LeakyReLU(lrelu_alpha)(convolve5)

        convolve5 = Conv2D(64 * mul, 3, padding='same')(convolve5)
        convolve5 = BatchNormalization(axis=1)(convolve5)
        convolve5 = LeakyReLU(lrelu_alpha)(convolve5)

        convolve5 = Conv2D(64 * mul, 3, padding='same')(convolve5)
        convolve5 = BatchNormalization(axis=1)(convolve5)
        convolve5 = LeakyReLU(lrelu_alpha)(convolve5)
        aux_outputs[4].append(convolve5)

    for i, o in enumerate(aux_outputs):
        if len(o) > 1:
            aux_outputs[i] = concatenate(o, axis=1)
        else:
            aux_outputs[i] = o[0]

    outputs = []
    layers = []
    for _n_out, _in in zip(n_out, inputs):
        convolve = Conv2DTranspose(32 * mul, 4, strides=2, padding='same')(aux_outputs[4])
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(32 * mul, 3, padding='same')(concatenate([aux_outputs[3], convolve], axis=1))
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(32 * mul, 3, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2DTranspose(16 * mul, 4, strides=2, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(16 * mul, 3, padding='same')(concatenate([aux_outputs[2], convolve], axis=1))
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(16 * mul, 3, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2DTranspose(8 * mul, 4, strides=2, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(8 * mul, 3, padding='same')(concatenate([aux_outputs[1], convolve], axis=1))
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(8 * mul, 3, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2DTranspose(4 * mul, 4, strides=2, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(4 * mul, 3, padding='same')(concatenate([aux_outputs[0], convolve], axis=1))
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(4 * mul, 3, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        convolve = LeakyReLU(lrelu_alpha)(convolve)

        convolve = Conv2D(_n_out, 1, padding='same')(convolve)
        convolve = BatchNormalization(axis=1)(convolve)
        # output = add([input_, convolve])
        # layer = Model(input_, output, name='U-net')
        layer = Add()
        layers.append(layer)
        output = layer([_in, convolve])
        outputs.append(output)
    if not return_list:
        layers = layers[0]
        outputs = outputs[0]
    return layers, outputs, aux_outputs


# def contrast_net(inputs, n_out=1, separate=False):
#     # if isinstance(inputs, list):
#     #     filters = [contrast_filter_block(_in) for _in in inputs]
#     #     filters = concatenate(filters)
#     # else:
#     #     filters = contrast_filter_block(inputs)
#     filters = contrast_filter_block(inputs)
#     # domain transfer
#     sz = MU['training'].shape[2]
#     if separate:
#         dense = [Dense(sz * sz)(filters) for _ in range(n_out)]
#         dense = [Reshape((1, sz, sz))(l) for l in dense]
#         outputs = [u_net(l) for l in dense]
#         for i in range(n_out):
#             outputs[i] = Activation(activation)(outputs[i])
#             outputs[i] = Lambda(fn)(outputs[i])
#         if n_out == 1:
#             return outputs[0]
#         else:
#             return outputs
#     else:
#         dense = Dense(sz * sz * n_out)(filters)
#         dense = Reshape((n_out, sz, sz))(dense)
#         outputs = u_net(dense)
#         outputs = Activation(activation)(outputs)
#         outputs = Lambda(fn)(outputs)
#         if n_out == 1:
#             return outputs
#         else:
#             return [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(outputs) for i in range(n_out)]
#     # dense = Reshape((64 * n_out, sz//8, sz//8))(dense)
#     # convolve = BatchNormalization(axis=1)(dense)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     # convolve = Conv2DTranspose(32 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(16 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(8 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(n_out, 5, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(n_out, 3, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # outputs = add([dense, convolve])
#     # convolve = Dropout(rate_2)(convolve)
#
#     # outputs = u_net(dense)
#     # outputs = Activation(activation)(outputs)
#     # outputs = Lambda(fn)(outputs)
#
#     # outputs = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(convolve) for i in range(n_out)]
#
#     # outputs = [u_net(o) for o in outputs]
#     # outputs = [u_net(concatenate(filters)) for _ in range(n_out)]
#
#     # for i in range(len(outputs)):
#     #     outputs[i] = Activation(activation)(outputs[i])
#     #     outputs[i] = Lambda(fn)(outputs[i])
#
#     # if n_out == 1:
#         # return outputs[0]
#         # return outputs
#     # else:
#         # return outputs
#         # return [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(outputs) for i in range(n_out)]


def background_filter_block(inputs, n_out=1):
    # n_out = int(inputs[0].shape[1])
    # outputs, freq_param, d_param = inputs
    # l = int(np.sqrt(K.int_shape(outputs)[-1]))
    #feature extraction
    # output = Conv1D(16, l, strides=l, data_format='channels_first')(outputs)
    # output = LeakyReLU(lrelu_alpha)(output)
    # output = Conv1D(16, 1, data_format='channels_first')(output)
    # output = LeakyReLU(lrelu_alpha)(output)
    # output = Conv1D(960 * n_out, 16, strides=16, data_format='channels_first')(output)

    # output2 = Conv1D(64 * n_out, PHI_meas['training'].shape[-1]//30, dilation_rate=15,
    #                  data_format='channels_first')(inputs)
    # output2 = LeakyReLU(lrelu_alpha)(output2)
    # output2 = Conv1D(60 * n_out, 1, data_format='channels_first')(output2)
    # output2 = LeakyReLU(lrelu_alpha)(output2)
    # output2 = Conv1D(960 * n_out, 15, strides=15, data_format='channels_first')(output2)
    outputs = []
    for _in in inputs[:-2]:
        output = Conv1D(16, 15, strides=15, activation='relu', data_format='channels_first')(_in)
        output = Conv1D(16, 1, activation='relu', data_format='channels_first')(output)
        output = GlobalAveragePooling1D(data_format='channels_first')(output)
        outputs.append(output)
    # if len(outputs) > 1:
    #     outputs = concatenate(outputs)
    # else:
    #     outputs = outputs[0]
    outputs = concatenate(outputs + inputs[-2:])
    # output = add([output, freq_param, d_param])
    # output = Flatten()(output)
    return outputs


def background_combine_mag_phs_block(inputs):
    # output = Dense(64)(inputs)
    # output = LeakyReLU(lrelu_alpha)(output)
    # output = Dropout(0.5)(output)
    output = Dense(16, activation='relu')(inputs)
    layer = Dense(1, activation=activation)
    output = layer(output)
    return layer, output


# def background_net(inputs, n_out=1):
#     if isinstance(inputs, list):
#         filters = [contrast_filter_block(_in) for _in in inputs]
#         filters = concatenate(filters)
#     else:
#         filters = contrast_filter_block(inputs)
#     # filters = background_filter_block(inputs)
#     outputs = [background_combine_mag_phs_block(filters) for _ in range(n_out)]
#     # outputs = [Dense(1, activation=activation)(filters) for _ in range(n_out)]
#     if n_out == 1:
#         return outputs[0]
#     else:
#         return outputs


# def background_and_contrast_net(inputs, units=None, n_out=1):
def background_and_contrast_net(inputs, input_mask=None, units=None, n_out=1):
    # if isinstance(inputs, list):
    #     filters = [contrast_filter_block(_in, input_mask) for _in in inputs]
    #     filters = concatenate(filters)
    # else:
    #     filters = contrast_filter_block(inputs, input_mask)
    # _, filters, bg_filters = contrast_filter_block(inputs, units)
    freq_param = Lambda(fn_freq, name='lambda_div_freq')(inputs[-2])
    d_param = Lambda(fn_d, name='lambda_div_d')(inputs[-1])
    if PREPROCESSING_MODE == 'SEQUENTIAL':
        meas = rnn_block(inputs[:-2])
    elif PREPROCESSING_MODE == 'FIXED':
        meas = inputs[:-2]
    else:
        raise NotImplementedError()
    filters = []
    bg_filters = []
    for i in range(n_out):
        _, f1, f2 = contrast_filter_block(meas + [freq_param, d_param], input_mask, units)
        # f1 = concatenate([f1, freq_param, d_param])
        # f2 = background_filter_block(inputs[:-2] + [freq_param, d_param])
        # f2 = concatenate([f2, freq_param, d_param])
        filters.append(f1)
        bg_filters.append(f2)
    # if n_out > 1:
    #     filters = concatenate(filters)
    #     bg_filters = concatenate(bg_filters)
    # else:
    #     filters = filters[0]
    #     bg_filters = bg_filters[0]
    # outputs = reshape_fixed_inputs(inputs, n_out=2)
    # _, filters, bg_filters = contrast_filter_block(inputs[:-2] + [freq_param, d_param], input_mask, units)
    # bg_filters = background_filter_block(inputs)
    # domain transfer
    sz = mask.shape[0]
    # contrast_layers = []
    contrasts = []
    background_layers = []
    backgrounds = []
    # aux = []
    for i in range(n_out):
        dense = Dense(sz * sz, kernel_initializer=initializers.RandomUniform(
            minval=0, maxval=np.sqrt(6/(K.int_shape(filters[i])[-1] + sz*sz))))(filters[i])
        dense = Reshape((1, sz, sz))(dense)
        dense = Lambda(fn_mask)(dense)
        contrasts.insert(0, dense)

        # l, o, _ = u_net(dense)
        # l.name = l.name + '_' + str(i)
        # o = Activation(activation)(o)
        # l = ReLU()
        # o = l(o)
        # contrast_layers.insert(0, l)
        # contrasts.insert(0, o)
        # contrast_layers.append(l)
        # contrasts.append(o)

        l, o = background_combine_mag_phs_block(bg_filters[i])
        background_layers.insert(0, l)
        backgrounds.insert(0, o)
        # background_layers.append(l)
        # backgrounds.append(o)
    if n_out > 1:
        contrasts = concatenate(contrasts, axis=1)
    else:
        contrasts = contrasts[0]

    # dense = Dense(sz * sz * n_out)(filters)
    # dense = Reshape((n_out, sz, sz))(dense)
    # contrasts = Lambda(fn)(contrasts)

    contrast_layers, contrasts, _ = u_net(contrasts, mul=2)
    # for i in range(n_out):
    #     contrast_layers[i] = ReLU()
    #     contrasts[i] = contrast_layers[i](contrasts[i])
    # contrasts = Activation(activation)(contrasts)
    # contrast_layers = Activation(activation)
    # contrast_layers = Lambda(fn)
    contrast_layers = ReLU()
    contrasts = contrast_layers(contrasts)
    # backgrounds = [Dense(1, activation=activation)(filters) for _ in range(n_out)]

    if n_out > 1:
        contrast_layers = []
        temp = []
        for i in range(n_out):
            l = Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)), name='lambda_slice_cnet_' + str(i))
            contrast_layers.append(l)
            temp.append(l(contrasts))
        contrasts = temp
    else:
        background_layers = background_layers[0]
        backgrounds = backgrounds[0]

    # if n_out == 1:
    #     contrast_layers = contrast_layers[0]
    #     contrasts = contrasts[0]
    #     background_layers = background_layers[0]
    #     backgrounds = backgrounds[0]

    return contrast_layers, background_layers, contrasts, backgrounds


def background_contrast_multiplier(inputs):
    # output0 = RepeatVector(np.prod(MU['training'].shape[2:]))(inputs[0])
    # output0 = Reshape((1,)+MU['training'].shape[2:])(output0)
    output0 = Reshape((1, 1, 1))(inputs[0])
    layer = Multiply()
    output = layer([inputs[1], output0])
    return layer, output


def denoiser1d_net(input_shape, separate=False):
    if isinstance(input_shape, int):
        input_shape = (1, input_shape)
    elif len(input_shape) == 1:
        input_shape = (1,) + tuple(input_shape)
    inputs = Input(input_shape)

    outputs = Conv1D(16, 3, padding='same', data_format='channels_first')(inputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(32, 3, padding='same', data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(64, 3, padding='same', data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(16, 1, data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    l = K.int_shape(outputs)[1:]
    outputs = Flatten()(outputs)
    outputs = Dense(128)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Dense(np.prod(l))(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)
    outputs = Reshape(l)(outputs)

    outputs = Conv1D(16, 1, data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(64, 3, padding='same', data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(32, 3, padding='same', data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    outputs = Conv1D(16, 3, padding='same', data_format='channels_first')(outputs)
    outputs = BatchNormalization(axis=1)(outputs)
    outputs = LeakyReLU(lrelu_alpha)(outputs)

    if separate:
        outputs = Conv1D(input_shape[0], 1, data_format='channels_first')(outputs)
        names = []
        temp = []
        for i in range(input_shape[0]):
            names.append('denoise_' + str(i))
            temp.append(Lambda(lambda x: K.slice(x, (0, i, 0), (-1, 1, -1)), name=names[i])(outputs))
        outputs = temp
    else:
        names = 'denoise'
        outputs = Conv1D(input_shape[0], 1, data_format='channels_first', name=names)(outputs)

    return Model(inputs=inputs, outputs=outputs), names


def primary_net(key):
    if PREPROCESSING_MODE == 'SEQUENTIAL':
        input_shape = (None,)+PHI_meas[key].shape[2:]
        inputs = [Input(input_shape), Input((1,)), Input((1,))]
        # input_mask = Lambda(lambda x: K.cast(K.not_equal(K.prod(x, -1), 0), dtype=K.floatx()))(inputs[0])
        # input_mask = Dropout(0.3)(input_mask)
        # input_mask = Reshape((-1, 1))(input_mask)
        # input_seq = multiply([inputs[0], input_mask])
        input_seq = inputs[0]
        input_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, 1)))(input_seq)
        input_phs = Lambda(lambda x: K.slice(x, (0, 0, 1), (-1, -1, 1)))(input_seq)
        input_loc = Lambda(lambda x: K.slice(x, (0, 0, 2), (-1, -1, input_shape[-1]-2)))(input_seq)
        input_mag = concatenate([input_mag, input_loc])
        input_phs = concatenate([input_phs, input_loc])
    elif PREPROCESSING_MODE == 'FIXED':
        input_shape = PHI_meas[key].shape[1:]
        inputs = [Input(input_shape), Input((1,)), Input((1,))]
        l = input_shape[-1]//2
        # input_mag = Lambda(lambda x: K.slice(x, (0, 0), (-1, l)))(inputs[0])
        # input_phs = Lambda(lambda x: K.slice(x, (0, l), (-1, l)))(inputs[0])
        input_ = Reshape((1,) + input_shape)(inputs[0])
        input_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l)))(input_)
        input_phs = Lambda(lambda x: K.slice(x, (0, 0, l), (-1, -1, l)))(input_)
    else:
        raise NotImplementedError()
    # l = PHI_meas['training'].shape[-1]
    # input_seq = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l-2)))(inputs[0])
    # input_loc = Lambda(lambda x: K.slice(x, (0, 0, l-2), (-1, -1, 2)))(inputs[0])
    # l_signal = (l-1)//2
    # input_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l_signal)))(inputs[0])
    # input_phs = Lambda(lambda x: K.slice(x, (0, 0, l_signal), (-1, -1, l_signal)))(inputs[0])
    # input_flag = Lambda(lambda x: K.cast(K.not_equal(x, 0), dtype=K.floatx()))(input_mag)
    # input_loc = Lambda(lambda x: K.slice(x, (0, 0, l-1), (-1, -1, 1)))(inputs[0])
    # input_mag2 = concatenate([input_mag, input_loc])
    # input_phs2 = concatenate([input_phs, input_loc])
    # input_mask = Lambda(lambda x: K.slice(x, (0, 0, l-1), (-1, -1, 1)))(inputs[0])
    # input_mask = Lambda(lambda x: K.cast(K.not_equal(K.sum(x, -1, keepdims=True), 0), dtype=K.floatx()))(inputs[0])
    # input_mask = Dropout(0.1)(input_mask)
    # input_seq = multiply([inputs[0], input_mask])
    # input_flag = multiply([input_flag, input_mask])
    # input_flag = Dropout(0.1)(input_flag)
    # input_mag = multiply([input_mag, input_flag])
    # input_phs = multiply([input_phs, input_flag])
    # input_seq = Dropout(0.1)(input_seq)
    # input_loc = Dropout(0.1)(input_loc)
    # input_seq = concatenate([input_seq, input_loc], axis=-1)
    # l = int(input1d.shape[2])//2
    # input1d_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l)))(input1d)
    # input1d_phs = Lambda(lambda x: K.slice(x, (0, 0, l), (-1, -1, l)))(input1d)
    # input1d = concatenate([input1d_mag, input1d_phs], axis=1)
    # contrast1 = contrast_net([input1d_mag, input1d_phs])
    # background1 = background_net([input1d_mag, input1d_phs])
    # contrast2 = contrast_net([input1d_mag, input1d_phs])
    # background2 = background_net([input1d_mag, input1d_phs])
    # contrast1, contrast2 = contrast_net([input1d_mag, input1d_phs], n_out=2)
    # background1, background2 = background_net([input1d_mag, input1d_phs], n_out=2)
    # contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
    #                                                      input_mask, rnn_units=[256], n_out=2)
    # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net(inputs, n_out=2)
    # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net(inputs, input_mask,
    #                                                                                          n_out=2)
    # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
    #                                                                                          input_mask, n_out=2)
    # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
    #                                                                                          n_out=2)
    contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net(
        [input_mag, input_phs] + inputs[1:], n_out=2)
    # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net(
    #     [input_mag, input_phs], n_out=2)
    contrast_layer1, contrast_layer2 = contrast_layers
    background_layer1, background_layer2 = background_layers
    contrast1, contrast2 = contrasts
    background1, background2 = backgrounds
    background_layer1 = Lambda(lambda x: x / 100, name='lambda_divide_constant_bg_MUa')
    background1 = background_layer1(background1)
    layer1, output1 = background_contrast_multiplier([background1, contrast1])
    layer2, output2 = background_contrast_multiplier([background2, contrast2])
    names = ['MUa_image', 'MUsp_image', 'MUa_contrast_image', 'MUsp_contrast_image',
             'MUa_background', 'MUsp_background']
    layer1.name = names[0]
    layer2.name = names[1]
    contrast_layer1.name = names[2]
    contrast_layer2.name = names[3]
    background_layer1.name = names[4]
    background_layer2.name = names[5]
    return Model(inputs=inputs, outputs=[output1, output2, contrast1, contrast2, background1, background2]), names


def validity_filter_block(inputs):
    output = Conv2D(16, 4, strides=2, padding='same')(inputs)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)

    output = Conv2D(32, 4, strides=2, padding='same')(output)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)

    output = Conv2D(64, 4, strides=2, padding='same')(output)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)

    output = Conv2D(64, 4, strides=2, padding='same')(output)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)

    output = Conv2D(64, 4, strides=2, padding='same')(output)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)

    output = Conv2D(64, 4, strides=2, padding='same')(output)
    output = BatchNormalization(axis=1)(output)
    output = LeakyReLU(lrelu_alpha)(output)
    # layer = Reshape((1, 64))
    layer = Flatten()
    output = layer(output)
    return layer, output


def secondary_net():
    inputs = [Input((1,)+MU['training_16'].shape[2:]), Input((1,)+MU['training_16'].shape[2:])]
    _, features1 = validity_filter_block(inputs[0])
    _, features2 = validity_filter_block(inputs[1])
    output = concatenate([features1, features2], axis=1)
    # output = Conv1D(1, 1, data_format='channels_first')(output)
    # output = Flatten()(output)
    # output1 = Dense(32)(output)
    # output1 = LeakyReLU(lrelu_alpha)(output1)
    # output2 = Dense(32)(output)
    # output2 = LeakyReLU(lrelu_alpha)(output2)

    # output = Dropout(rate_3)(output)
    # pos_output1 = Dense(2, activation=activation)(output1)
    # pos_output2 = Dense(2, activation=activation)(output2)
    # pos_layer = Concatenate()
    # area_output1 = Dense(1, activation=activation)(output1)
    # area_output2 = Dense(1, activation=activation)(output2)
    # area_layer = Concatenate()
    layer = Dense(1, activation='sigmoid')
    # pos_output = pos_layer([pos_output1, pos_output2])
    # area_output = area_layer([area_output1, area_output2])
    output = layer(output)
    names = ['discriminator_result', 'position', 'area']
    layer.name = names[0]
    # pos_layer.name = names[1]
    # area_layer.name = names[2]
    # output = Dense(16)(output)
    # output = BatchNormalization()(output)
    # output = Dense(1)(output)
    # return Model(inputs=inputs, outputs=[output, pos_output, area_output]), names
    return Model(inputs=inputs, outputs=output), [names[0]]

####
# def contrast_filter_block(inputs, input_mask=None):
# # def contrast_filter_block(inputs):
#     # n_out = int(inputs.shape[1])
#     # n_out = 1
#     #filtering
#     # output = Conv1D(16, 3, padding='same', data_format='channels_first')(inputs)
#     # output = Conv1D(16, 3, padding='same', data_format='channels_first')(output)
#     # output = ActivityRegularization(l1=reg_l1, l2=reg_l2)(output)
#     # output = BatchNormalization(axis=1)(output)
#
#     #skip connection
#     # _output = Conv1D(16, 3, padding='same', data_format='channels_first')(output)
#     # _output = BatchNormalization(axis=1)(_output)
#     # _output = LeakyReLU(lrelu_alpha)(_output)
#     #
#     # _output = Conv1D(16, 3, padding='same', data_format='channels_first')(_output)
#     # _output = BatchNormalization(axis=1)(_output)
#     # _output = LeakyReLU(lrelu_alpha)(_output)
#     #
#     # _output = Conv1D(16, 3, padding='same', data_format='channels_first')(_output)
#     # output = add([output, _output])
#     # _output = BatchNormalization(axis=1)(_output)
#     # output = LeakyReLU(lrelu_alpha)(output)
#     # output = SpatialDropout1D(rate_1)(output)
#
#     freq_param = Lambda(lambda x: x/100e6)(inputs[1])
#     d_param = Lambda(lambda x: x/100)(inputs[2])
#     # input_mask = Lambda(lambda x: K.squeeze(x, axis=2))(input_mask)
#     # outputs = Bidirectional(GRU(256))(inputs[0], mask=input_mask)
#     input_mask = Masking()(input_mask)
#     outputs = concatenate([inputs[0], input_mask])
#     outputs = Bidirectional(GRU(256))(outputs)
#     outputs = concatenate([outputs, freq_param, d_param])
#
#     #feature extraction
#     # output1 = Conv1D(64 * n_out, 15, strides=15, data_format='channels_first')(inputs)
#     # output1 = BatchNormalization(axis=1)(output1)
#     # output1 = LeakyReLU(lrelu_alpha)(output1)
#
#     # output1 = Conv1D(64 * n_out, 1, data_format='channels_first')(output1)
#     # output1 = BatchNormalization(axis=1)(output1)
#     # output1 = LeakyReLU(lrelu_alpha)(output1)
#
#     # output1 = Conv1D(64 * n_out, 1, data_format='channels_first')(output1)
#     # output1 = BatchNormalization(axis=1)(output1)
#     # output1 = LeakyReLU(lrelu_alpha)(output1)
#     # output1 = [Lambda(lambda x: K.slice(x, (0, 60 * i, 0), (-1, 60, -1)))(output1) for i in range(n_out)]
#     # output1 = [Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(o) for o in output1]
#     # output1 = concatenate(output1, axis=1)
#     # output1 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(output1)
#
#     # output1 = Conv1D(32 * n_out, 4, strides=2, padding='same', data_format='channels_first')(output1)
#     # output1 = BatchNormalization(axis=1)(output1)
#     # output1 = LeakyReLU(lrelu_alpha)(output1)
#
#     # output1 = Conv1D(64 * n_out, 4, strides=2, padding='same', data_format='channels_first')(output1)
#     # output1 = BatchNormalization(axis=1)(output1)
#     # output1 = LeakyReLU(lrelu_alpha)(output1)
#
#     # output2 = Conv1D(64 * n_out, PHI_meas['training'].shape[-1]//30, dilation_rate=15,
#     #                  data_format='channels_first')(inputs)
#     # output2 = BatchNormalization(axis=1)(output2)
#     # output2 = LeakyReLU(lrelu_alpha)(output2)
#
#     # output2 = Conv1D(64 * n_out, 1, data_format='channels_first')(output2)
#     # output2 = BatchNormalization(axis=1)(output2)
#     # output2 = LeakyReLU(lrelu_alpha)(output2)
#
#     # output2 = Conv1D(64 * n_out, 1, data_format='channels_first')(output2)
#     # output2 = BatchNormalization(axis=1)(output2)
#     # output2 = LeakyReLU(lrelu_alpha)(output2)
#
#     # output = add([output1, output2])
#     # output = BatchNormalization(axis=1)(output)
#     # output = LeakyReLU(lrelu_alpha)(output)
#     # output = Flatten()(output1)
#     return outputs
#
#
# def u_net(inputs):
#     n_out = int(inputs.shape[1])
#     mul = 1
#     # dense = Dense(np.prod(MU['training'].shape[2:]))(inputs)
#     # dense = Reshape((1,) + MU['training'].shape[2:])(dense)
#
#     # dense = Reshape((64, int(MU['training'].shape[2]/8), int(MU['training'].shape[3]/8)))(dense)
#     #
#     # convolve = Conv2DTranspose(32, 4, strides=2, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(16, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(8, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(4, 1, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(2, 1, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(1, 1, padding='same')(convolve)
#     # # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     # convolve = Lambda(fn)(convolve)
#
#     # skip connection
#     convolve1 = Conv2D(4 * mul, 3, padding='same')(inputs)
#     convolve1 = BatchNormalization(axis=1)(convolve1)
#     convolve1 = LeakyReLU(lrelu_alpha)(convolve1)
#
#     convolve1 = Conv2D(4 * mul, 3, padding='same')(convolve1)
#     convolve1 = BatchNormalization(axis=1)(convolve1)
#     convolve1 = LeakyReLU(lrelu_alpha)(convolve1)
#
#     convolve2 = Conv2D(8 * mul, 4, strides=2, padding='same')(convolve1)
#     convolve2 = BatchNormalization(axis=1)(convolve2)
#     convolve2 = LeakyReLU(lrelu_alpha)(convolve2)
#
#     convolve2 = Conv2D(8 * mul, 3, padding='same')(convolve2)
#     convolve2 = BatchNormalization(axis=1)(convolve2)
#     convolve2 = LeakyReLU(lrelu_alpha)(convolve2)
#
#     convolve2 = Conv2D(8 * mul, 3, padding='same')(convolve2)
#     convolve2 = BatchNormalization(axis=1)(convolve2)
#     convolve2 = LeakyReLU(lrelu_alpha)(convolve2)
#
#     convolve3 = Conv2D(16 * mul, 4, strides=2, padding='same')(convolve2)
#     convolve3 = BatchNormalization(axis=1)(convolve3)
#     convolve3 = LeakyReLU(lrelu_alpha)(convolve3)
#
#     convolve3 = Conv2D(16 * mul, 3, padding='same')(convolve3)
#     convolve3 = BatchNormalization(axis=1)(convolve3)
#     convolve3 = LeakyReLU(lrelu_alpha)(convolve3)
#
#     convolve3 = Conv2D(16 * mul, 3, padding='same')(convolve3)
#     convolve3 = BatchNormalization(axis=1)(convolve3)
#     convolve3 = LeakyReLU(lrelu_alpha)(convolve3)
#
#     convolve4 = Conv2D(32 * mul, 4, strides=2, padding='same')(convolve3)
#     convolve4 = BatchNormalization(axis=1)(convolve4)
#     convolve4 = LeakyReLU(lrelu_alpha)(convolve4)
#
#     convolve4 = Conv2D(32 * mul, 3, padding='same')(convolve4)
#     convolve4 = BatchNormalization(axis=1)(convolve4)
#     convolve4 = LeakyReLU(lrelu_alpha)(convolve4)
#
#     convolve4 = Conv2D(32 * mul, 3, padding='same')(convolve4)
#     convolve4 = BatchNormalization(axis=1)(convolve4)
#     convolve4 = LeakyReLU(lrelu_alpha)(convolve4)
#
#     convolve5 = Conv2D(64 * mul, 4, strides=2, padding='same')(convolve4)
#     convolve5 = BatchNormalization(axis=1)(convolve5)
#     convolve5 = LeakyReLU(lrelu_alpha)(convolve5)
#
#     convolve5 = Conv2D(64 * mul, 3, padding='same')(convolve5)
#     convolve5 = BatchNormalization(axis=1)(convolve5)
#     convolve5 = LeakyReLU(lrelu_alpha)(convolve5)
#
#     convolve5 = Conv2D(64 * mul, 3, padding='same')(convolve5)
#     convolve5 = BatchNormalization(axis=1)(convolve5)
#     convolve5 = LeakyReLU(lrelu_alpha)(convolve5)
#
#     convolve = Conv2DTranspose(32 * mul, 4, strides=2, padding='same')(convolve5)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(32 * mul, 3, padding='same')(concatenate([convolve4, convolve], axis=1))
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(32 * mul, 3, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2DTranspose(16 * mul, 4, strides=2, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(16 * mul, 3, padding='same')(concatenate([convolve3, convolve], axis=1))
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(16 * mul, 3, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2DTranspose(8 * mul, 4, strides=2, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(8 * mul, 3, padding='same')(concatenate([convolve2, convolve], axis=1))
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(8 * mul, 3, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2DTranspose(4 * mul, 4, strides=2, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(4 * mul, 3, padding='same')(concatenate([convolve1, convolve], axis=1))
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(4 * mul, 3, padding='same')(convolve)
#     convolve = BatchNormalization(axis=1)(convolve)
#     convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     convolve = Conv2D(n_out, 1, padding='same')(convolve)
#     output = add([inputs, convolve])
#     return output
#
#
# def contrast_net(inputs, n_out=1, separate=False):
#     if isinstance(inputs, list):
#         filters = [contrast_filter_block(_in) for _in in inputs]
#         filters = concatenate(filters)
#     else:
#         filters = contrast_filter_block(inputs)
#     # domain transfer
#     sz = MU['training'].shape[2]
#     if separate:
#         dense = [Dense(sz * sz)(filters) for _ in range(n_out)]
#         dense = [Reshape((1, sz, sz))(l) for l in dense]
#         outputs = [u_net(l) for l in dense]
#         for i in range(n_out):
#             outputs[i] = Activation('softplus')(outputs[i])
#             outputs[i] = Lambda(fn)(outputs[i])
#         if n_out == 1:
#             return outputs[0]
#         else:
#             return outputs
#     else:
#         dense = Dense(sz * sz * n_out)(filters)
#         dense = Reshape((n_out, sz, sz))(dense)
#         outputs = u_net(dense)
#         outputs = Activation('softplus')(outputs)
#         outputs = Lambda(fn)(outputs)
#         if n_out == 1:
#             return outputs
#         else:
#             return [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(outputs) for i in range(n_out)]
#     # dense = Reshape((64 * n_out, sz//8, sz//8))(dense)
#     # convolve = BatchNormalization(axis=1)(dense)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     # convolve = Conv2DTranspose(32 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(16 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2DTranspose(8 * n_out, 4, strides=2, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(n_out, 5, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(16, 3, padding='same')(dense)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # convolve = LeakyReLU(lrelu_alpha)(convolve)
#     #
#     # convolve = Conv2D(n_out, 3, padding='same')(convolve)
#     # convolve = BatchNormalization(axis=1)(convolve)
#     # outputs = add([dense, convolve])
#     # convolve = Dropout(rate_2)(convolve)
#
#     # outputs = u_net(dense)
#     # outputs = Activation('softplus')(outputs)
#     # outputs = Lambda(fn)(outputs)
#
#     # outputs = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(convolve) for i in range(n_out)]
#
#     # outputs = [u_net(o) for o in outputs]
#     # outputs = [u_net(concatenate(filters)) for _ in range(n_out)]
#
#     # for i in range(len(outputs)):
#     #     outputs[i] = Activation('softplus')(outputs[i])
#     #     outputs[i] = Lambda(fn)(outputs[i])
#
#     # if n_out == 1:
#         # return outputs[0]
#         # return outputs
#     # else:
#         # return outputs
#         # return [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(outputs) for i in range(n_out)]
#
#
# def background_filter_block(inputs):
#     n_out = int(inputs.shape[1])
#     #feature extraction
#     output = Conv1D(16 * n_out, 15, strides=15, data_format='channels_first')(inputs)
#     output = LeakyReLU(lrelu_alpha)(output)
#     output = Conv1D(16 * n_out, 1, data_format='channels_first')(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#     # output = Conv1D(960 * n_out, 16, strides=16, data_format='channels_first')(output)
#
#     # output2 = Conv1D(64 * n_out, PHI_meas['training'].shape[-1]//30, dilation_rate=15,
#     #                  data_format='channels_first')(inputs)
#     # output2 = LeakyReLU(lrelu_alpha)(output2)
#     # output2 = Conv1D(60 * n_out, 1, data_format='channels_first')(output2)
#     # output2 = LeakyReLU(lrelu_alpha)(output2)
#     # output2 = Conv1D(960 * n_out, 15, strides=15, data_format='channels_first')(output2)
#     output = GlobalAveragePooling1D(data_format='channels_first')(output)
#     # output = add([output, output2])
#     # output = Flatten()(output)
#     return output
#
#
# # def background_combine_mag_phs_block(inputs):
# #     output = Dense(64)(inputs)
# #     output = LeakyReLU(lrelu_alpha)(output)
# #     output = Dense(1, activation='softplus')(output)
# #     return output
#
#
# def background_net(inputs, n_out=1):
#     if isinstance(inputs, list):
#         filters = [contrast_filter_block(_in) for _in in inputs]
#         filters = concatenate(filters)
#     else:
#         filters = contrast_filter_block(inputs)
#     # filters = background_filter_block(inputs)
#     outputs = [background_combine_mag_phs_block(filters) for _ in range(n_out)]
#     # outputs = [Dense(1, activation='softplus')(filters) for _ in range(n_out)]
#     if n_out == 1:
#         return outputs[0]
#     else:
#         return outputs
#
#
# def background_and_contrast_net(inputs, input_mask=None, n_out=1):
# # def background_and_contrast_net(inputs, n_out=1):
#     # if isinstance(inputs, list):
#     #     filters = [contrast_filter_block(_in) for _in in inputs]
#     #     filters = concatenate(filters)
#     # else:
#     #     filters = contrast_filter_block(inputs)
#     filters = contrast_filter_block(inputs, input_mask)
#     # filters = contrast_filter_block(inputs)
#     # domain transfer
#     sz = MU['training'].shape[2]
#     dense = Dense(sz * sz * n_out)(filters)
#     dense = Reshape((n_out, sz, sz))(dense)
#
#     contrasts = u_net(dense)
#     contrasts = Activation(activation)(contrasts)
#     contrast_layers = Lambda(fn)
#     contrasts = contrast_layers(contrasts)
#     # backgrounds = [Dense(1, activation='softplus')(filters) for _ in range(n_out)]
#     # backgrounds = [background_combine_mag_phs_block(filters)[1] for _ in range(n_out)]
#     background_layers = []
#     backgrounds = []
#     for _ in range(n_out):
#         l, o = background_combine_mag_phs_block(filters)
#         background_layers.append(l)
#         backgrounds.append(o)
#
#     if n_out > 1:
#         # contrasts = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))(contrasts) for i in range(n_out)]
#         # contrast_layers = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1))) for i in range(n_out)]
#         # contrasts = [l(contrasts) for l in contrast_layers]
#         contrast_layers = []
#         temp = []
#         for i in range(n_out):
#             l = Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1)))
#             contrast_layers.append(l)
#             temp.append(l(contrasts))
#         contrasts = temp
#     else:
#         background_layers = background_layers[0]
#         backgrounds = backgrounds[0]
#
#     return contrasts, backgrounds
# #
# #     if n_out > 1:
# #         contrast_layers = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1))) for i in range(n_out)]
# #         contrasts = [l(contrasts) for l in contrast_layers]
# #     else:
# #         background_layers = background_layers[0]
# #         backgrounds = backgrounds[0]
# #
# #     return contrast_layers, background_layers, contrasts, backgrounds
#
#
#
# # def background_and_contrast_net(inputs, input_mask=None, units=None, n_out=1):
# #     # if isinstance(inputs, list):
# #     #     filters = [contrast_filter_block(_in, input_mask) for _in in inputs]
# #     #     filters = concatenate(filters)
# #     # else:
# #     #     filters = contrast_filter_block(inputs, input_mask)
# #     # _, filters, bg_filter = contrast_filter_block(inputs, units)
# #     _, filters, bg_filter = contrast_filter_block(inputs, input_mask, units)
# #     # domain transfer
# #     sz = MU['training'].shape[2]
# #     dense = Dense(sz * sz * n_out)(filters)
# #     dense = Reshape((n_out, sz, sz))(dense)
# #
# #     _, contrasts = u_net(dense)
# #     contrasts = Activation(activation)(contrasts)
# #     # contrast_layers = Activation(activation)
# #     contrast_layers = Lambda(fn)
# #     contrasts = contrast_layers(contrasts)
# #     # backgrounds = [Dense(1, activation=activation)(filters) for _ in range(n_out)]
# #     background_layers = []
# #     backgrounds = []
# #     for _ in range(n_out):
# #         l, o = background_combine_mag_phs_block(bg_filter)
# #         background_layers.append(l)
# #         backgrounds.append(o)
# #
# #     if n_out > 1:
# #         contrast_layers = [Lambda(lambda x: K.slice(x, (0, i, 0, 0), (-1, 1, -1, -1))) for i in range(n_out)]
# #         contrasts = [l(contrasts) for l in contrast_layers]
# #     else:
# #         background_layers = background_layers[0]
# #         backgrounds = backgrounds[0]
# #
# #     return contrast_layers, background_layers, contrasts, backgrounds
#
#
# def background_contrast_multiplier(inputs):
#     output0 = RepeatVector(np.prod(MU['training'].shape[2:]))(inputs[0])
#     output0 = Reshape((1,)+MU['training'].shape[2:])(output0)
#     output = multiply([inputs[1], output0])
#     return output
#
#
# def primary_net():
#     inputs = [Input((None,)+PHI_meas['training'].shape[2:]), Input((1,)), Input((1,))]
#     l = PHI_meas['training'].shape[-1]
#     input_seq = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l-1)))(inputs[0])
#     input_mask = Lambda(lambda x: K.slice(x, (0, 0, l-1), (-1, -1, 1)))(inputs[0])
#     input_mask = Dropout(0.1)(input_mask)
#     # input0 = concatenate([input_seq, input_mask], axis=2)
#     # input1d = Reshape((1, PHI_meas['training'].shape[1]))(inputs)
#     # l = int(input1d.shape[2])//2
#     # input1d_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l)))(input1d)
#     # input1d_phs = Lambda(lambda x: K.slice(x, (0, 0, l), (-1, -1, l)))(input1d)
#     # input1d = concatenate([input1d_mag, input1d_phs], axis=1)
#     # contrast1 = contrast_net([input1d_mag, input1d_phs])
#     # background1 = background_net([input1d_mag, input1d_phs])
#     # contrast2 = contrast_net([input1d_mag, input1d_phs])
#     # background2 = background_net([input1d_mag, input1d_phs])
#     # contrast1, contrast2 = contrast_net([input1d_mag, input1d_phs], n_out=2)
#     # background1, background2 = background_net([input1d_mag, input1d_phs], n_out=2)
#     contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:], input_mask, n_out=2)
#     # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
#     #                                                                                          input_mask, n_out=2)
#     # contrasts, backgrounds = background_and_contrast_net([input0] + inputs[1:], n_out=2)
#     contrast1, contrast2 = contrasts
#     background1, background2 = backgrounds
#     background1 = Lambda(lambda x: x / 100)(background1)
#     output1 = background_contrast_multiplier([background1, contrast1])
#     output2 = background_contrast_multiplier([background2, contrast2])
#     names = ['MUa_image', 'MUsp_image', 'MUa_contrast_image', 'MUsp_contrast_image',
#              'MUa_background', 'MUsp_background']
#     return Model(inputs=inputs, outputs=[output1, output2, contrast1, contrast2, background1, background2]), names
#
#
# # def primary_net():
# #     inputs = [Input((None,)+PHI_meas['training'].shape[2:]), Input((1,)), Input((1,))]
# #     l = PHI_meas['training'].shape[-1]
# #     input_seq = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l-1)))(inputs[0])
# #     # l_signal = (l-3)//2
# #     # input_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l_signal)))(inputs[0])
# #     # input_phs = Lambda(lambda x: K.slice(x, (0, 0, l_signal), (-1, -1, l_signal)))(inputs[0])
# #     # input_loc = Lambda(lambda x: K.slice(x, (0, 0, l-3), (-1, -1, 2)))(inputs[0])
# #     # input_mag2 = concatenate([input_mag, input_loc])
# #     # input_phs2 = concatenate([input_phs, input_loc])
# #     input_mask = Lambda(lambda x: K.slice(x, (0, 0, l-1), (-1, -1, 1)))(inputs[0])
# #     input_mask = Dropout(0.1)(input_mask)
# #     # l = int(input1d.shape[2])//2
# #     # input1d_mag = Lambda(lambda x: K.slice(x, (0, 0, 0), (-1, -1, l)))(input1d)
# #     # input1d_phs = Lambda(lambda x: K.slice(x, (0, 0, l), (-1, -1, l)))(input1d)
# #     # input1d = concatenate([input1d_mag, input1d_phs], axis=1)
# #     # contrast1 = contrast_net([input1d_mag, input1d_phs])
# #     # background1 = background_net([input1d_mag, input1d_phs])
# #     # contrast2 = contrast_net([input1d_mag, input1d_phs])
# #     # background2 = background_net([input1d_mag, input1d_phs])
# #     # contrast1, contrast2 = contrast_net([input1d_mag, input1d_phs], n_out=2)
# #     # background1, background2 = background_net([input1d_mag, input1d_phs], n_out=2)
# #     # contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
# #     #                                                      input_mask, rnn_units=[256], n_out=2)
# #     # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net(inputs, n_out=2)
# #     contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
# #                                                                                              input_mask, n_out=2)
# #     # contrast_layers, background_layers, contrasts, backgrounds = background_and_contrast_net([input_seq] + inputs[1:],
# #     #                                                                                          n_out=2)
# #     contrast_layer1, contrast_layer2 = contrast_layers
# #     background_layer1, background_layer2 = background_layers
# #     contrast1, contrast2 = contrasts
# #     background1, background2 = backgrounds
# #     background_layer1 = Lambda(lambda x: x / 100)
# #     background1 = background_layer1(background1)
# #     layer1, output1 = background_contrast_multiplier([background1, contrast1])
# #     layer2, output2 = background_contrast_multiplier([background2, contrast2])
# #     names = ['MUa_image', 'MUsp_image', 'MUa_contrast_image', 'MUsp_contrast_image',
# #              'MUa_background', 'MUsp_background']
# #     layer1.name = names[0]
# #     layer2.name = names[1]
# #     contrast_layer1.name = names[2]
# #     contrast_layer2.name = names[3]
# #     background_layer1.name = names[4]
# #     background_layer2.name = names[5]
# #     return Model(inputs=inputs, outputs=[output1, output2, contrast1, contrast2, background1, background2]), names
#
#
# def validity_filter_block(inputs):
#     output = Conv2D(16, 4, strides=2, padding='same')(inputs)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#
#     output = Conv2D(32, 4, strides=2, padding='same')(output)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#
#     output = Conv2D(64, 4, strides=2, padding='same')(output)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#
#     output = Conv2D(64, 4, strides=2, padding='same')(output)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#
#     output = Conv2D(64, 4, strides=2, padding='same')(output)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#
#     output = Conv2D(64, 4, strides=2, padding='same')(output)
#     output = BatchNormalization(axis=1)(output)
#     output = LeakyReLU(lrelu_alpha)(output)
#     output = Flatten()(output)
#     return output
#
#
# def secondary_net():
#     inputs = [Input((1,)+MU['training'].shape[2:]), Input((1,)+MU['training'].shape[2:])]
#     features1 = validity_filter_block(inputs[0])
#     features2 = validity_filter_block(inputs[1])
#     output = concatenate([features1, features2])
#     # output = Dropout(rate_3)(output)
#     output = Dense(1, activation='sigmoid')(output)
#     # output = Dense(16)(output)
#     # output = BatchNormalization()(output)
#     # output = Dense(1)(output)
#     return Model(inputs=inputs, outputs=output)