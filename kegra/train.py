from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(batch_shape=(None,None))]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X] + T_k
    G = [Input(batch_shape=(None,None)) for _ in range(support)]

else:
    raise ValueError('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), weighted_metrics=['accuracy'])

class Logger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(logs["loss"]),
          "train_acc= {:.4f}".format(logs["accuracy"]),
          "val_loss= {:.4f}".format(logs["val_loss"]),
          "val_acc= {:.4f}".format(logs["val_accuracy"]),
        )

# training iteration
history = model.fit(
    graph,
    y_train,
    sample_weight=train_mask,
    batch_size=A.shape[0],
    epochs=NB_EPOCH,
    shuffle=False,
    verbose=0,
    validation_data=(graph, y_val, val_mask),
    callbacks=[EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    ), Logger()]
)

# Testing
test_loss, test_acc = model.evaluate(graph, 
    y_test, 
    sample_weight=test_mask, 
    batch_size=A.shape[0],
    verbose=0)
print("Test set results:",
      "loss= {:.4f}".format(test_loss),
      "accuracy= {:.4f}".format(test_acc))
