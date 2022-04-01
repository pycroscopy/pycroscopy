import os
import shutil
import numpy as np
from scipy import signal, io
from argparse import ArgumentParser, Namespace
import h5py
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_tqdm import TQDMCallback
from keras.models import Sequential
from keras.layers import (Dense, Conv1D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D)


def load_data(data_path, resample_size=256):
    f = h5py.File(data_path, 'r')
    loop_data = f['filt_AI_mat'][()]
    X = np.rollaxis(loop_data.reshape(loop_data.shape[0], -1), 1)
    X_resample = np.zeros((X.shape[0], 40 * resample_size))
    for i in range(X.shape[0]):
        X_resample[i] = signal.resample(X[i], 40 * resample_size)
    X_resample = X_resample.reshape((-1, resample_size))
    X_resample -= np.mean(X_resample)  # TODO remove mean of each singal?
    X_resample /= np.std(X_resample)
    X_resample = np.atleast_3d(X_resample)

    return X_resample


def get_run_id(sample, layer_type, size, num_layers, embedding, lr, drop_frac, batch_size, kernel_size, **kwargs):
    run = (f"{sample}_{layer_type}{size:03d}_x{num_layers}_emb{embedding:03d}_{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}_batch{batch_size}").replace('e-', 'm')
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    return run


def rnn_auto(layer, size, num_layers, embedding, n_step, drop_frac=0., bidirectional=True,
             **kwargs):
    if bidirectional:
        wrapper = Bidirectional
    else:
        wrapper = lambda x: x
    model = Sequential()
    model.add(wrapper(layer(size, return_sequences=(num_layers > 1)),
                        input_shape=(n_step, 1)))
    for i in range(1, num_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_layers - 1), dropout=drop_frac)))
    model.add(Dense(embedding, activation='linear', name='encoding'))
    model.add(RepeatVector(n_step))
    for i in range(num_layers):
        model.add(wrapper(layer(size, return_sequences=True, dropout=drop_frac)))
    model.add(TimeDistributed(Dense(1, activation='linear')))

    return model


def conv_auto(size, num_layers, embedding, n_step, kernel_size, drop_frac=0., **kwargs):
    """TODO: batch norm?"""
    model = Sequential()
    model.add(Conv1D(size, kernel_size, padding='same', activation='relu',
                     input_shape=(n_step, 1)))
    for i in range(1, num_layers):
        model.add(MaxPooling1D(2))
        model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))

    model.add(Dense(embedding, activation='linear', name='encoding'))
#    model.add(Conv1D(embedding, kernel_size=1, activation='linear', name='encoding'))

    for i in range(1, num_layers):
        model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))
        model.add(UpSampling1D(2))
    model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))

    return model


def main(arg_dict=None):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/cleaned_data.mat')
    parser.add_argument("--size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument('--embedding', type=int)
    parser.add_argument("--drop_frac", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_cycles", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--layer_type", type=str)
    parser.add_argument("--N_train", type=int)
    parser.add_argument("--kernel_size", type=int, default=5)
#    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--cache', type=bool, default=False)
    parser.add_argument('--sample', type=str)
    parser.set_defaults(bidirectional=True, overwrite=False)
    args = parser.parse_args(None if arg_dict is None else [])  # don't read argv if arg_dict present
    if arg_dict:  # merge additional arguments w/ defaults
        args = Namespace(**{**args.__dict__, **arg_dict})


    # TODO fix the problems with the NANs
    if args.sample in ['PZT_2080_BEPS', 'PZT_Mixed_BEPS']:
        data = io.matlab.loadmat('Data.mat')
        if args.sample in ['PZT_2080_BEPS']:
            X = data['Loopdata_caca'].reshape(-1, 64)

            # Sets the nan values = 0 TODO improve with in interpolator
            X[np.where(np.isnan(X))] = 0
            # subtracts the mean and takes the standard deviation of the data
            X -= np.mean(X)
            X /= np.std(X)
            X = np.atleast_3d(X)
        elif args.sample in ['PZT_Mixed_BEPS']:
            X = data['Loopdata_mixed'].reshape(-1, 96)

            # Sets the nan values = 0 TODO improve with in interpolator
            X[np.where(np.isnan(X))] = 0

            # subtracts the mean and takes the standard deviation of the data
            X -= np.mean(X)
            X /= np.std(X)
            X = np.atleast_3d(X)

    elif args.sample in ['PZT_Mixed_GEPS']:
        if not args.cache:
            X = load_data(args.data_path, args.n_cycles)
        else:
            X = np.load('data/X_256.npy')
            if args.n_cycles != 256:
                raise ValueError("256 hard coded")
    else:
        raise ValueError('Name of file does not exist')

    if args.N_train:
        train = np.arange(args.N_train)
    else:
        train = np.arange(len(X))

    print(X.shape)
    run = get_run_id(**args.__dict__)
    log_dir = os.path.join(args.log_dir, run)
    print("Logging to {}".format(os.path.abspath(log_dir)))
    weights_path = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(weights_path):
        if args.overwrite:
            print(f"Overwriting {log_dir}")
            shutil.rmtree(log_dir, ignore_errors=True)
        else:
            raise ValueError("Model file already exists")

    layer = {'lstm': LSTM, 'gru': GRU, 'conv': Conv1D}[args.layer_type]
    if issubclass(layer, Recurrent):
        model = rnn_auto(layer, args.size, args.num_layers, args.embedding, n_step=X.shape[1],
                         drop_frac=args.drop_frac)
    elif issubclass(layer, Conv1D):
#def conv_auto(size, num_layers, embedding, n_step, kernel_size, drop_frac=0., **kwargs):
        model = conv_auto(args.size, args.num_layers, args.embedding, n_step=X.shape[1],
                          kernel_size=args.kernel_size, drop_frac=args.drop_frac)
    else:
        raise NotImplementedError()
    model.compile(Adam(args.lr), loss='mse')

    history = model.fit(X[train], X[train], epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=[TQDMCallback(),
                                   TensorBoard(log_dir=log_dir, write_graph=False),
                                   ModelCheckpoint(weights_path)],
                        verbose=False)

    return X, model, history


if __name__ == '__main__':
    X, model, history = main()
