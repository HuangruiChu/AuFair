from abc import ABC, abstractmethod
import tensorflow as tf
from codebase.mlp import MLP
tf.compat.v1.disable_eager_execution()
import numpy as np
# defaults
EPS = 1e-12
HIDDEN_LAYER_SPECS =  {'layer_sizes': [5], 'activ': 'sigmoid'}
CLASS_COEFF = 1.
FAIR_COEFF = 0.
PASS_COEFF = 0.
XDIM = 21
YDIM = 1
ADIM = 1
SEED = 0
Y2DIM = 1

class AbstractBaseNet(ABC):
    def __init__(self,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 **kwargs):
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff
        self.xdim = xdim
        self.ydim = ydim
        self.adim = adim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        tf.compat.v1.set_random_seed(self.seed)
        self._define_vars()

    @abstractmethod
    def _define_vars(self):
        pass

    @abstractmethod
    def _get_class_logits(self, scope_name='model/preds'):  
        pass

    @abstractmethod
    def _get_class_preds(self, scope_name='model/preds'): 
        pass

    @abstractmethod
    def _get_class_loss(self):
        pass

    @abstractmethod
    def _get_fairness_regularizer(self): 
        pass

    @abstractmethod
    def _get_loss(self): 
        pass

class BinaryMLP(AbstractBaseNet):

    def __init__(self,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 **kwargs):

        super().__init__(class_coeff, fair_coeff, xdim, ydim, adim, hidden_layer_specs, seed=seed)
        self.Y_hat_logits = self._get_class_logits()
        self.Y_hat = self._get_class_preds()
        self.class_loss = self._get_class_loss()
        self.fair_reg = self._get_fairness_regularizer()
        self.loss = self._get_loss()
        self.idk_loss = tf.zeros_like(self.class_loss)
        self.idks = tf.zeros_like(self.Y_hat)
        self.loss_class = self._get_class_loss()
        self.loss_idk = tf.zeros_like(self.loss_class)
        self.Y_ttl = self.Y_hat

    def _define_vars(self):
        self.X = tf.compat.v1.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.compat.v1.placeholder("float", [None, self.ydim], name='Y')
        self.A = tf.compat.v1.placeholder("float", [None, self.adim], name='A')
        self.Y_DM = tf.compat.v1.placeholder("float", [None, self.ydim], name='Y_DM')
        self.epoch = tf.compat.v1.placeholder("float", [1], name='epoch')
        return

    def _get_class_logits(self, scope_name='model/preds'):
        with tf.compat.v1.variable_scope(scope_name):
            mlp = MLP(name='data_to_class_preds',
                      shapes=[self.xdim] + self.hidden_layer_specs['layer_sizes'] + [self.ydim],
                      activ=self.hidden_layer_specs['activ'])
            logits = mlp.forward(self.X)
            return logits

    def _get_class_preds(self, scope_name='model/preds'):
        preds = tf.nn.sigmoid(self.Y_hat_logits)
        return preds

    def _get_class_loss(self):
        return cross_entropy(self.Y, self.Y_hat)

    def _get_fairness_regularizer(self):
        return di_regularizer(self.Y, self.A, self.Y_hat)

    def _get_loss(self):
        return self.class_coeff * tf.reduce_mean(self.class_loss) + self.fair_coeff * self.fair_reg

class IDKModel(AbstractBaseNet):

    def __init__(self,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 pass_coeff=PASS_COEFF,
                 **kwargs):

        super().__init__(class_coeff, fair_coeff, xdim, ydim, adim, hidden_layer_specs, seed=seed)
        self.pass_coeff = pass_coeff
        self.Y_hat_logits = self._get_class_logits()
        self.Y_hat = self._get_class_preds()
        self.idks = self._get_idks()
        self.class_loss = self._get_class_loss()
        self.fair_reg = self._get_fairness_regularizer()
        self.idk_loss = self._get_idk_loss()
        self.loss = self._get_loss()
        self.Y_ttl = hard_switch(self.Y_hat, self.Y_DM, self.idks)

    @abstractmethod
    def _get_idks(self): #weight class loss for final loss function
        pass

    @abstractmethod
    def _get_idk_loss(self): #weight recon loss for final loss function
        pass

class MLPRejectModel(IDKModel):
    def _define_vars(self):
        self.X = tf.compat.v1.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.compat.v1.placeholder("float", [None, self.ydim], name='Y')
        self.A = tf.compat.v1.placeholder("float", [None, self.adim], name='A')
        self.Y_DM = tf.compat.v1.placeholder("float", [None, self.ydim], name='Y_DM')
        self.epoch = tf.compat.v1.placeholder("float", [1], name='epoch')
        return

    def _get_class_logits(self, scope_name='model/preds'):
        with tf.compat.v1.variable_scope(scope_name):
            mlp = MLP(name='data_to_class_preds',
                      shapes=[self.xdim] + self.hidden_layer_specs['layer_sizes'] + [self.ydim],
                      activ=self.hidden_layer_specs['activ'])
            logits = mlp.forward(self.X)
            return logits

    def _get_class_preds(self, scope_name='model/preds'):
        preds = tf.sigmoid(self.Y_hat_logits)
        return preds

    def _get_class_loss(self):
        ce = cross_entropy(self.Y, self.Y_hat)
        return ce

    def _get_fairness_regularizer(self):
        return di_regularizer(self.Y, self.A, self.Y_hat)

    def _get_idks(self, scope_name='model/preds'):
        #
        with tf.compat.v1.variable_scope(scope_name):
            mlp = MLP(name='data_to_idks',
                      shapes=[self.xdim + self.ydim] + self.hidden_layer_specs['layer_sizes'] + [self.ydim], 
                      activ=self.hidden_layer_specs['activ'])
            Y_hat_no_grad = tf.stop_gradient(self.Y_hat)
            self.idk_logits = mlp.forward(tf.concat([self.X, Y_hat_no_grad], axis=1))
            idks = tf.sigmoid(self.idk_logits) 
            return idks

    def _get_idk_loss(self):
        return self.idks 

    def _get_loss(self):
        self.loss_class = self.class_coeff * tf.reduce_mean(self.class_loss)
        idk_weighted_class_loss = tf.multiply(1. - self.idks, tf.expand_dims(self.class_loss, 1))
        self.loss_idk = self.pass_coeff * tf.reduce_mean(self.idk_loss)
        return self.class_coeff * tf.reduce_mean(idk_weighted_class_loss) \
               + self.pass_coeff * tf.reduce_mean(self.idk_loss) \
                + self.fair_coeff * self.fair_reg

class MLPDeferModel(MLPRejectModel):
    def _get_class_loss(self):
        CE_model = tf.expand_dims(cross_entropy(self.Y, self.Y_hat), 1)
        #class error for Decision maker, 但这个Y_DM 从哪里得到？data 里面有
        CE_DM = tf.expand_dims(cross_entropy(self.Y, self.Y_DM), 1)
        cl = tf.squeeze(switch(CE_model, CE_DM, self.idks), axis=[1])
        return cl

    def _get_loss(self):
        class_loss_no_grad = tf.expand_dims(self.class_loss, 1)
        class_loss_idk = tf.multiply(self.idks, class_loss_no_grad)
        self.loss_class = self.class_coeff * tf.reduce_mean(self.class_loss)
        self.loss_idk = self.pass_coeff * tf.reduce_mean(self.idk_loss) \
                - self.class_coeff * tf.reduce_mean(class_loss_idk)
        return self.class_coeff * tf.reduce_mean(self.class_loss) \
               + self.pass_coeff * tf.reduce_mean(self.idk_loss) \
                + self.fair_coeff * self.fair_reg 

    def _get_fairness_regularizer(self):
        #Concrete relaxation/Gumbel-Softmax
        temperature = 0.5
        gumbel_idks = gumbel_binary_sample(self.idks, temperature)
        sampled_Y  = switch(self.Y_hat, self.Y_DM, gumbel_idks)
        di_reg = di_regularizer(self.Y, self.A, sampled_Y)
        return di_reg


def cross_entropy(target, pred, eps=EPS):
    l = -tf.squeeze(tf.multiply(target, tf.math.log(pred + eps)) + tf.multiply(1 - target, tf.math.log(1 - pred + eps)), axis=[1])
    return l

def di_regularizer(Y, A, Y_hat):
    # 0: unprivileged, 1: privileged
    # Y: ground truth, A: protected attribute, Y_hat: prediction
    # fpr: false positive rate, fnr: false negative rate
    # tpr: true positive rate, tnr: true negative rate
    #original Equalized Odds 
    # fpr0 = soft_rate(1 - Y, 1 - A, Y_hat)
    # fpr1 = soft_rate(1 - Y, A, Y_hat)
    # fnr0 = soft_rate(Y, 1 - A, Y_hat)
    # fnr1 = soft_rate(Y, A, Y_hat)

    # fpdi = tf.abs(fpr0 - fpr1)
    # fndi = tf.abs(fnr0 - fnr1)

    # di = 0.5 * (fpdi + fndi)
    #change to Equal Opportunity Difference
    #which is the difference of true positive rate
    # Y_hat = Y_hat.numpy()
    # Y = Y.numpy()
    # A = A.numpy()
    # TP = tf.multiply(Y, Y_hat)
    # mask0 = tf.multiply(Y,1-A)
    # mask1 = tf.multiply(Y,A)
    # TP0 = tf.multiply(TP, mask0)
    # TP1 = tf.multiply(TP, mask1)
    # tpr0 = tf.reduce_sum(TP0) / (tf.reduce_sum(mask0) + EPS)
    # tpr1 = tf.reduce_sum(TP1) / (tf.reduce_sum(mask1) + EPS)
    tpr0 = soft_rate(Y, 1 - A, Y_hat)
    tpr1 = soft_rate(Y, A, Y_hat)
    di = abs(tpr1 - tpr0)

    return di

def soft_rate(ind1, ind2, pred): 
    mask = tf.multiply(ind1, ind2)
    rate = tf.reduce_sum(tf.multiply(tf.abs(pred - ind1), mask)) / tf.reduce_sum(mask + EPS)
    return rate

def switch(x0, x1, s):
    return tf.multiply(x0, 1. - s) + tf.multiply(x1, s)

def hard_switch(x0, x1, s):
    s_ind = tf.cast(tf.greater(s, 0.5), tf.float32)
    return tf.multiply(x0, 1. - s_ind) + tf.multiply(x1, s_ind)

#Code from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

def sample_gumbel(shape, eps=EPS):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_binary_sample(logits, t):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    gumbel_noise_on = sample_gumbel(tf.shape(logits))
    gumbel_noise_off = sample_gumbel(tf.shape(logits))
    concrete_on = (tf.math.log(logits + EPS) + gumbel_noise_on) / t
    concrete_off = (tf.math.log(1 - logits + EPS) + gumbel_noise_off) / t
    concrete_softmax = tf.divide(tf.exp(concrete_on), tf.exp(concrete_on) + tf.exp(concrete_off))
    return concrete_softmax

