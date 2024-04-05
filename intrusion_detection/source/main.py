import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

import pandas as pd
from typing import List, Tuple

import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

def seed_packages(seed_val=666):
    os.environ['PYTHONHASHSEED'] = str(seed_val)

    # NOTE
    #   > calling set_random_seed(seed_val) is equivalent to
    #       calling
    #           * random.seed(seed_val)
    #           * np.random.set_seed(seed_val)
    #           * tf.random.set_seed(seed_val)
    #           * torch.manual_seed(seed_val)
    tf.keras.utils.set_random_seed(seed_val)

    # target CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # renounce on parallel execution
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # determinism for gpu devices - does it really work?
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def make_deterministic(seed_val=666):
    seed_packages(seed_val)
    tf.config.experimental.enable_op_determinism()


def load_data() -> pd.DataFrame:
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
                 "urgent", "hot", "num_failed_logins", "logged_in",
                 "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                 "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                 "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                 "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "label"]

    df = pd.read_csv("data/KDDTrain+_20Percent.txt", header=None, names=col_names, index_col=False)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # transform categorical data
    categorical_variables = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    categorical_data = pd.get_dummies(df[categorical_variables])

    numerical_variables = list(set(df.columns.values.tolist()) - set(categorical_variables))
    numerical_variables.sort()
    numerical_variables.remove('label')
    numerical_data = df[numerical_variables].copy()

    df_preprocessed = pd.concat([numerical_data, categorical_data], axis=1)
    return df_preprocessed


def split_data(df_orig: pd.DataFrame, df_preprocessed: pd.DataFrame, test_size=0.25):
    labels = df_orig['label'].copy()
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(df_preprocessed,
                                                        integer_labels,
                                                        test_size=test_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def build_autoencoder_model(input_dim: int, num_neurons_per_layer_list: List[int], activation_func: str,
                            latent_space_dim: int, seed_val=666) -> Model:
    # input layer
    input_data = Input(shape=(input_dim,), name='encoder_input')

    # hidden layers of encoder
    num_hidden_layers = len(num_neurons_per_layer_list)
    encoder = Dense(units=num_neurons_per_layer_list[0], activation=activation_func, name='encoder_0',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                    bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(input_data)
    for i in range(1, num_hidden_layers):
        encoder = Dense(units=num_neurons_per_layer_list[i], activation=activation_func,
                        name='encoder_{:d}'.format(i),
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                        bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(encoder)

    # bottleneck layer
    latent_encoding = Dense(latent_space_dim, activation='linear', name='latent_encoding',
                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                            bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(encoder)

    # hidden layers of decoder
    decoder = Dense(units=num_neurons_per_layer_list[num_hidden_layers - 1], activation=activation_func,
                    name='decoder_0',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                    bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(latent_encoding)
    for i in range(1, num_hidden_layers):
        decoder = Dense(units=num_neurons_per_layer_list[num_hidden_layers - 1 - i], activation=activation_func,
                        name='decoder_{:d}'.format(i),
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                        bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(decoder)

    # output layer
    reconstructed_data = Dense(units=input_dim, activation='linear', name='reconstructed_data',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val),
                               bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed_val))(decoder)

    autoencoder_model = Model(input_data, reconstructed_data)
    return autoencoder_model


def compile_autoencoder_model(model: Model, learning_rate: float, loss_function: str):
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)


def train_autoencoder_model(model: Model, x_train, x_test, num_epochs: int, batch_size: int) -> History:
    history = model.fit(x_train, x_train, shuffle=True, epochs=num_epochs, batch_size=batch_size,
                        validation_data=(x_test, x_test))
    return history


def generate_anomaly_labels(labels) -> List[int]:
    anomaly_labels = []
    for item in labels:
        is_anomaly = (item != 11)
        anomaly_labels.append(int(is_anomaly))
    return anomaly_labels


def plot_roc_curve(fpr, tpr):
    auc_roc = auc(fpr, tpr)     # area under curve

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)'.format(auc_roc))
    ax.plot([0, 1], [0, 1], color='lime', linestyle='--')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()


def compute_accuracy(mtrx: np.ndarray) -> float:
    acc = -1.0
    num_samples = mtrx.sum()
    if num_samples > 0:
        acc = mtrx.trace() / num_samples
    return acc


def evaluate_autoencoder_model(model: Model, x_eval, y_eval) -> Tuple[float, float]:
    data_reconstructed = model.predict(x_eval)
    reconstruction_scores = np.mean((x_eval - data_reconstructed) ** 2, axis=1)

    anomaly_labels = generate_anomaly_labels(y_eval)
            # anomaly_data = pd.DataFrame({'reconstruction_scores': reconstruction_scores, 'anomaly_labels': anomaly_labels})

    # ### generate roc curve and find optimal threshold
    fpr, tpr, thresholds = roc_curve(anomaly_labels, reconstruction_scores)
    plot_roc_curve(fpr, tpr)

    optimal_threshold_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_idx]

    pred_labels = (reconstruction_scores > optimal_threshold).astype(int)

    cnfsn_mtrx = confusion_matrix(anomaly_labels, pred_labels)
    plot_confusion_matrix(cnfsn_mtrx, ['normal', 'anomaly'])
    accuracy = compute_accuracy(cnfsn_mtrx)

    return optimal_threshold, accuracy


def visualise_model(model: Model):
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='LR')


def visualise_training_history(history: History):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.legend(['loss on train data', 'loss on validation data'])
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix, target_names):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img_plot = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix')
    fig.colorbar(img_plot, ax=ax, orientation='vertical', fraction=.1)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks, target_names, rotation=45)
    ax.set_yticks(tick_marks, target_names)
    fig.tight_layout()

    width, height = confusion_matrix.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(confusion_matrix[x][y]), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    plt.show()
    plt.close(fig)

def main():
    make_deterministic()

    # ### load and preprocess data
    data_df = load_data()
    data_df_preprocessed = preprocess_data(data_df)
    x_train, x_test, y_train, y_test = split_data(data_df, data_df_preprocessed, test_size=0.25)

    # ### transform and convert data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = x_train.astype(np.float32)

    x_test = scaler.transform(x_test)
    x_test = x_test.astype(np.float32)

    # ### build autoencoder model
    input_space_dim = x_train.shape[1]
    latent_space_dim = 5
    neurons_per_layer_list = [16, 48, 64, 96]
    activation_func = 'relu'
    autoencoder_model = build_autoencoder_model(input_space_dim, neurons_per_layer_list, activation_func,
                                                latent_space_dim)

    # ### visualise model
    visualise_model(autoencoder_model)

    # ### compile model
    learning_rate = 1e-4
    loss_function = 'mse'
    compile_autoencoder_model(autoencoder_model, learning_rate, loss_function)

    # ### train model
    num_epochs = 11
    batch_size = 512
    history = train_autoencoder_model(autoencoder_model, x_train, x_test, num_epochs, batch_size)

    ### visualise training and validation loss
    visualise_training_history(history)

    ### evaluate training result
    threshold, accuracy = evaluate_autoencoder_model(autoencoder_model, x_test, y_test)

    print(' > threshold = {:.4f}'.format(threshold))
    print(' > accuracy = {:.4f}'.format(accuracy))



if __name__ == "__main__":
    main()
