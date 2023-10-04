"""
Custom optimizers to implement lr_mult as in caffe

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

References
----------
https://github.com/keras-team/keras/issues/5920#issuecomment-328890905
"""


from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

class customSGD(optimizers.SGD):
    """
    Custom subclass of the SGD optimizer to implement lr_mult as in Caffe
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, lr_mult=0.1, excluded_vars=[], **kwargs):
        super().__init__(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [self.iterations.assign_add(1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        # Momentum
        moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + moments

        for p, g, m in zip(params, grads, moments):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr = lr
            else:
                multiplied_lr = lr * self.lr_mult
            ###################################################

            v = self.momentum * m - multiplied_lr * g  # Velocity
            self.updates.append(m.assign(v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(p.assign(new_p))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'excluded_vars': self.excluded_vars,
            'lr_mult': self.lr_mult
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class customAdam(optimizers.Adam):
    """
    Custom subclass of the Adam optimizer to implement lr_mult as in Caffe
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False,
                 lr_mult=0.1, excluded_vars=[], **kwargs):
        super().__init__(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [self.iterations.assign_add(1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
              1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1. - K.pow(self.beta_2, t)) /
            (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr_t = lr_t
            else:
                multiplied_lr_t = lr_t * self.lr_mult
            ###################################################

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - multiplied_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(vhat.assign(vhat_t))
            else:
                p_t = p - multiplied_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(m.assign(m_t))
            self.updates.append(v.assign(v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(p.assign(new_p))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'excluded_vars': self.excluded_vars,
            'lr_mult': self.lr_mult
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))





class customAdamW(Optimizer):
    """
    Custom subclass of the AdamW optimizer.
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.025,
                 batch_size=1, samples_per_epoch=1, epochs=1,
                 lr_mult=0.1, excluded_vars=[], **kwargs):
        super(customAdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = lr
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.decay = decay
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.samples_per_epoch = samples_per_epoch
            self.epochs = epochs
            self.lr_mult = lr_mult
            self.excluded_vars = excluded_vars
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [self.iterations.assign_add(1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            ####################################################
            # Add a lr multiplier for vars outside excluded_vars
            if p.name in self.excluded_vars:
                multiplied_lr_t = lr_t
            else:
                multiplied_lr_t = lr_t * self.lr_mult
            ###################################################

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            eta_t = 1.
            p_t = p - eta_t * (multiplied_lr_t * m_t / (K.sqrt(v_t) + self.epsilon))

            if self.weight_decay != 0:
                w_d = self.weight_decay * K.sqrt(self.batch_size / (self.samples_per_epoch * self.epochs))
                p_t = p_t - eta_t * (w_d * p)

            self.updates.append(m.assign(m_t))
            self.updates.append(v.assign(v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(p.assign(new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': self.lr,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'decay': self.decay,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'samples_per_epoch': self.samples_per_epoch,
            'epochs': self.epochs,
            'epsilon': self.epsilon,
            'excluded_vars': self.excluded_vars,
            'lr_mult': self.lr_mult
        }
        base_config = super(customAdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



