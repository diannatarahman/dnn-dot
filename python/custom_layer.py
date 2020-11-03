from keras.layers import *
from keras import backend as K, initializers, activations
import numpy as np


class AccumulatorCell(Layer):

    def __init__(self, locs, channels, units=None, div=None, max_output=256, **kwargs):
        self.locs = locs
        self.units = units
        if self.units is None:
            self.units = self.locs
            self.div = div
            if self.div is None:
                self.div = 1
        else:
            self.div = int(np.ceil(self.locs / self.units))
        self.channels = channels
        # self.activation = activations.get(activation)
        self.max_output = max_output
        if self.max_output is not None:
            self.compute_div()
        # self.state_size = [self.units ** 2 for _ in range(2)]
        self.state_size = [self.units ** 2 for _ in range(self.channels)] + [1]
        # self.state_size.insert(-1, 1)
        # self.state_size += [self.units ** 2 for _ in range(self.channels + 1)]
        # self.state_size.insert(-1, 1)
        # self.state_size += [self.units ** 2, 1]
        # self.state_size += [self.locs for _ in range(2)]
        super(AccumulatorCell, self).__init__(**kwargs)

    def compute_div(self):
        if self.units ** 2 > self.max_output:
            max_units = int(np.sqrt(self.max_output))
            self.div = int(np.ceil(self.locs / max_units))
            self.units = int(np.ceil(self.locs / self.div))

    def build(self, input_shape):
        # input_dim = input_shape[-1] - 2 + 2 * self.channels
        # input_dim_signal = input_shape[-1]
        # input_dim = self.channels + 2*self.units ** 2
        # self.embeddings0 = self.add_weight(shape=(self.locs + 1, 1), initializer='uniform', name='embeddings0')
        # self.embeddings1 = self.add_weight(shape=(self.locs + 1, 1), initializer='uniform', name='embeddings1')
        # self.kernel_signal = self.add_weight(shape=(input_dim_signal, self.channels), initializer='glorot_uniform',
        #                                      name='kernel_signal')
        # self.bias_signal = self.add_weight(shape=(self.channels,), initializer='zeros', name='bias_signal')
        # self.kernel = self.add_weight(shape=(input_dim, self.units ** 2), initializer='glorot_uniform', name='kernel')
        # self.bias = self.add_weight(shape=(self.units ** 2,), initializer='zeros', name='bias')
        # self.embeddings0 = self.add_weight(shape=(self.locs, self.channels), initializer='uniform', name='embeddings0')
        # self.embeddings1 = self.add_weight(shape=(self.locs, self.channels), initializer='uniform', name='embeddings1')
        # self.kernel0 = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='kernel0')
        # self.bias0 = self.add_weight(shape=(self.units,), initializer='zeros', name='bias0')
        # self.kernel1 = self.add_weight(shape=(input_dim, self.channels), initializer='glorot_uniform', name='kernel')
        # self.bias1 = self.add_weight(shape=(self.channels,), initializer='zeros', name='bias')
        c = np.linspace(0, self.locs, self.units, endpoint=False)
        c0 = np.kron(c, np.ones_like(c))
        c1 = (c0 + np.kron(np.ones_like(c), c)) % self.locs
        self.c0 = K.constant(c0.astype('int32') + 1)
        self.c1 = K.constant(c1.astype('int32') + 1)
        self.a0 = self.add_weight(shape=(1,), initializer=initializers.Constant(value=10.0), name='a0')
        self.a1 = self.add_weight(shape=(1,), initializer=initializers.Constant(value=10.0), name='a1')
        self.built = True

    def call(self, inputs, states):
        # prev_outputs = K.concatenate([K.expand_dims(s, axis=1) for s in states], axis=1)
        # soft_output = states[-2]
        prev_outputs = K.concatenate([K.expand_dims(s, axis=1) for s in states[:self.channels]], axis=1)
        counter = states[self.channels]
        # counter_rbf = states[self.channels + 1]
        # prev_outputs_src = K.concatenate([K.expand_dims(s, axis=1) for s in
        #                                   states[(self.channels + 2):(self.channels * 2 + 2)]], axis=1)
        # counter_src = states[self.channels * 2 + 2]
        # counter_rbf_src = states[self.channels * 2 + 3]
        # prev_loc0 = states[self.channels * 2 + 5]
        # gate = states[self.channels + 2]
        # counter_rbf0 = states[self.channels + 2]
        # counter_rbf1 = states[self.channels + 3]
        # counter_rbf01 = states[self.channels + 4]
        # flag0 = K.cast(states[self.channels + 5], 'bool')
        # flag1 = K.cast(states[self.channels + 6], 'bool')
        l = int(inputs.shape[1])
        loc = K.slice(inputs, (0, l - 2), (-1, 2))
        # loc_int = K.cast(loc, 'int32')
        loc0 = K.slice(loc, (0, 0), (-1, 1))
        loc1 = K.slice(loc, (0, 1), (-1, 1))
        signal = K.slice(inputs, (0, 0), (-1, l - 2))
        # embed_loc0 = K.gather(self.embeddings0, K.cast(K.flatten(loc0), 'int32'))
        # embed_loc1 = K.gather(self.embeddings1, K.cast(K.flatten(loc1), 'int32'))
        # in_h = K.concatenate([signal, embed_loc0, embed_loc1])
        # # h = K.dot(in_h, self.kernel0)
        # # h = K.bias_add(h, self.bias0, data_format='channels_last')
        # # h = activations.softmax(h)
        # h1 = K.dot(in_h, self.kernel1)
        # h1 = K.bias_add(h1, self.bias1, data_format='channels_last')
        # # if self.activation is not None:
        # #     h = self.activation(h)
        # alpha0 = K.log(1 + K.exp(self.a0))
        # alpha1 = K.log(1 + K.exp(self.a1))
        valid = K.all(K.greater(loc, 0), axis=1, keepdims=True)
        valid_floatx = K.cast(valid, K.floatx())
        _180deg = self.locs // 2
        dist0 = 1 - K.abs(K.abs(loc0 - self.c0) / _180deg - 1)
        dist1 = 1 - K.abs(K.abs(loc1 - self.c1) / _180deg - 1)
        # dist01 = 1 - K.abs(K.abs(loc0 - self.c1) / _180deg - 1)
        rbf0 = K.exp(-self.a0 * dist0**2)
        rbf1 = K.exp(-self.a1 * dist1**2) * valid_floatx
        # rbf01 = rbf0 * K.exp(-self.a1 * dist01**2)
        rbf = rbf0 * rbf1
        # change_src = K.equal(loc0, prev_loc0) | ~valid
        # change_src_floatx = K.cast(change_src, K.floatx())
        # not_change_src_floatx = K.cast(~change_src, K.floatx())
        # embed_loc0 = K.gather(self.embeddings0, K.cast(K.flatten(loc0), 'int32'))
        # embed_loc1 = K.gather(self.embeddings1, K.cast(K.flatten(loc1), 'int32'))
        # in_h = K.concatenate([signal, embed_loc0, embed_loc1])
        # h = K.dot(in_h, self.kernel_signal)
        # h = K.bias_add(h, self.bias_signal, data_format='channels_last')
        # h = activations.sigmoid(h)
        # in_gate = K.concatenate([h, rbf, gate])
        # gate = K.dot(in_gate, self.kernel)
        # gate = K.bias_add(gate, self.bias, data_format='channels_last')
        # gate = activations.sigmoid(gate)
        # one_hot0 = K.cast(K.one_hot(K.cast(K.flatten(loc0 - 1), 'int32'), self.locs), 'bool')
        # valid_flag0 = ~K.any(one_hot0 & flag0, axis=1, keepdims=True) & valid
        # one_hot1 = K.cast(K.one_hot(K.cast(K.flatten(loc1 - 1), 'int32'), self.locs), 'bool')
        # valid_flag1 = ~K.any(one_hot1 & flag1, axis=1, keepdims=True) & valid
        # flag0 |= one_hot0 & valid_flag0
        # flag1 |= one_hot1 & valid_flag1
        # one_hot0 = K.one_hot((loc0 - 1) // self.div, self.units)
        # one_hot0 = K.permute_dimensions(one_hot0, (0, 2, 1))
        # one_hot1 = K.one_hot((loc1 - 1) // self.div, self.units)
        # one_hot = one_hot0 * one_hot1
        # one_hot = K.reshape(one_hot, (-1, np.prod(one_hot.shape[1:])))
        # counter += one_hot * valid
        # h = one_hot0 * K.expand_dims(h, 1)
        # h = K.reshape(h, (-1, np.prod(h.shape[1:])))
        # soft_output += h * valid
        # outputs = K.expand_dims(one_hot, 1) * K.expand_dims(h1 * valid)
        # div = K.clip(counter_src, 1, None)
        # outputs = K.expand_dims(rbf0, 1) * prev_outputs_src / div
        # outputs = outputs * change_src_floatx + prev_outputs
        # outputs_src = K.expand_dims(rbf1, 1) * K.expand_dims(signal)
        # outputs_src += prev_outputs_src * not_change_src_floatx
        # counter_rbf += rbf0 * counter_rbf_src / div * change_src_floatx
        # counter_rbf_src = counter_rbf_src * not_change_src_floatx + rbf1
        # counter += valid_floatx * change_src_floatx
        # counter_src = counter_src * not_change_src_floatx + valid_floatx
        # feed_loc0 = loc0 * change_src_floatx + prev_loc0 * not_change_src_floatx
        outputs = K.expand_dims(rbf, 1) * K.expand_dims(signal)
        outputs += prev_outputs
        # counter_rbf += rbf
        counter += valid_floatx
        # valid_flag0_floatx = K.cast(valid_flag0, K.floatx())
        # counter_rbf0 += rbf0 * valid_flag0_floatx
        # counter_rbf1 += rbf1 * K.cast(valid_flag1, K.floatx())
        # counter_rbf01 += rbf01 * valid_flag0_floatx
        next_states = []
        for i in range(self.channels):
            o = K.slice(outputs, (0, i, 0), (-1, 1, -1))
            o = K.squeeze(o, 1)
            if i == 0:
                output = o
            next_states.append(o)
        next_states.append(counter)
        # next_states.append(counter_rbf)
        # for i in range(self.channels):
        #     o = K.slice(outputs_src, (0, i, 0), (-1, 1, -1))
        #     o = K.squeeze(o, 1)
        #     next_states.append(o)
        # next_states.append(counter_src)
        # next_states.append(counter_rbf_src)
        # next_states.append(rbf0)
        # next_states.append(feed_loc0)
        # output = outputs
        # output = K.squeeze(output, 1)
        # next_states.append(output)
        # next_states.append(rbf)
        # next_states.append(dist0)
        # next_states.append(dist1)
        # next_states.append(soft_output)
        # next_states.append(gate)
        # next_states.append(counter_rbf0)
        # next_states.append(counter_rbf1)
        # next_states.append(counter_rbf01)
        # next_states.append(K.cast(flag0, K.floatx()))
        # next_states.append(K.cast(flag1, K.floatx()))
        return output, next_states

    def get_config(self):
        config = {'locs': self.locs,
                  'units': self.units,
                  'div': self.div,
                  'channels': self.channels,
                  # 'activation': activations.serialize(self.activation),
                  'max_output': self.max_output}
        base_config = super(AccumulatorCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
