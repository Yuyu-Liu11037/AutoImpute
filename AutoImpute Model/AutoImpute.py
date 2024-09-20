import os
import argparse
import datetime
import matplotlib
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='AutoImpute')

# Print debug statements
parser.add_argument('--debug', type=bool, default=True, nargs='+', help="Want debug statements.")
parser.add_argument('--debug_display_step', type=int, default=1, help="Display loss after.")

# Hyper-parameters
parser.add_argument('--hidden_units', type=int, default=2000, help="Size of hidden layer or latent space dimensions.")
parser.add_argument('--lambda_val', type=int, default=1,
                    help="Regularization coefficient, to control the contribution of regularization term in the cost function.")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="Initial value of learning rate.")
parser.add_argument('--iterations', type=int, default=7000, help="Number of iterations to train the model for.")
parser.add_argument('--threshold', type=int, default=0.0001,
                    help="To stop gradient descent after the change in loss function value in consecutive iterations is less than the threshold, implying convergence.")

# Data
parser.add_argument('--data', type=str, default='blakeley.csv',
                    help="Dataset to run the script on.")

# Run the masked matrix recovery test
parser.add_argument('--masked_matrix_test', type=bool, default=False, nargs='+',
                    help="Run the masked matrix recovery test?")
parser.add_argument('--masking_percentage', type=float, default=10, nargs='+',
                    help="Percentage of masking required. Like 10, 20, 12.5 etc")

# Model save and restore options
parser.add_argument('--save_model_location', type=str, default='checkpoints/model1.ckpt',
                    help="Location to save the learnt model")
parser.add_argument('--load_model_location', type=str, default='checkpoints/model0.ckpt',
                    help="Load the saved model from.")
parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# masked and imputed matrix save location / name
parser.add_argument('--imputed_save', type=str, default='imputed_matrix', help="save the imputed matrix as")
parser.add_argument('--masked_save', type=str, default='masked_matrix', help="save the masked matrix as")

FLAGS = parser.parse_args()

if __name__ == '__main__':
    # started clock
    start_time = datetime.datetime.now()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if FLAGS.debug:
        if not FLAGS.load_saved:
            with open(FLAGS.log_file, 'w') as log:
                log.write('Step\tLoss\tLoss per Cell\t Change \n')

    # reading dataset
    try:
        extn = FLAGS.data.split('.')[1]
        if extn == 'mat':
            print("[!data read] Reading from data/" + FLAGS.data)
            processed_count_matrix = scipy.io.mmread("data/" + FLAGS.data)
            processed_count_matrix = processed_count_matrix.toarray()
            processed_count_matrix = np.array(processed_count_matrix)
        else:
            print("[!data read] Reading from ./data/" + FLAGS.data)
            with open("./data/" + FLAGS.data) as f:
                ncols = len(f.readline().split(','))
            processed_count_matrix = np.loadtxt(open("./data/" + FLAGS.data, "rb"), delimiter=",", skiprows=1, usecols=range(1, ncols))
    except Exception as e:
        print(f"[!data read] Error: {e}")
        exit()

    dataset = FLAGS.data.split('.')[0]

    if FLAGS.masked_matrix_test:
        masking_percentage = FLAGS.masking_percentage
        masking_percentage = masking_percentage / 100.0

        idxi, idxj = np.nonzero(processed_count_matrix)

        ix = np.random.choice(len(idxi), int(np.floor(masking_percentage * len(idxi))), replace=False)
        store_for_future = processed_count_matrix[idxi[ix], idxj[ix]]
        indices = idxi[ix], idxj[ix]

        processed_count_matrix[idxi[ix], idxj[ix]] = 0  # making masks 0
        matrix_mask = processed_count_matrix.copy()
        matrix_mask[matrix_mask.nonzero()] = 1

        if FLAGS.masked_save:
            scipy.io.savemat(FLAGS.masked_save + str(masking_percentage * 100) + ".mat",
                             mdict={"arr": processed_count_matrix})

        mae = []
        rmse = []
        nmse = []

    # finding number of genes and cells.
    genes = processed_count_matrix.shape[1]
    cells = processed_count_matrix.shape[0]
    print(f"[info] Genes : {genes}, Cells : {cells}")

    # Model definition with TensorFlow 2.x
    matrix_mask = processed_count_matrix.copy()
    matrix_mask[matrix_mask.nonzero()] = 1
    X = tf.convert_to_tensor(processed_count_matrix, dtype=tf.float32)
    mask = tf.convert_to_tensor(matrix_mask, dtype=tf.float32)

    weights = {
        'encoder_h': tf.Variable(tf.random.normal([genes, FLAGS.hidden_units])),
        'decoder_h': tf.Variable(tf.random.normal([FLAGS.hidden_units, genes])),
    }
    biases = {
        'encoder_b': tf.Variable(tf.random.normal([FLAGS.hidden_units])),
        'decoder_b': tf.Variable(tf.random.normal([genes])),
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h']), biases['encoder_b']))
        return layer_1

    def decoder(x):
        layer_1 = tf.add(tf.matmul(x, weights['decoder_h']), biases['decoder_b'])
        return layer_1

    # Loss and optimization with TensorFlow 2.x
    optimizer = tf.optimizers.RMSprop(FLAGS.initial_learning_rate)

    @tf.function
    def compute_loss(X, mask):
        encoded = encoder(X)
        decoded = decoder(encoded)
        rmse_loss = tf.reduce_sum(tf.square((X - decoded) * mask))
        regularization = (FLAGS.lambda_val / 2.0) * (
            tf.reduce_sum(tf.square(weights['decoder_h'])) +
            tf.reduce_sum(tf.square(weights['encoder_h']))
        )
        total_loss = rmse_loss + regularization
        return total_loss

    @tf.function
    def train_step(X, mask):
        with tf.GradientTape() as tape:
            loss = compute_loss(X, mask)
        grads = tape.gradient(loss, [weights['encoder_h'], weights['decoder_h'], biases['encoder_b'], biases['decoder_b']])
        optimizer.apply_gradients(zip(grads, [weights['encoder_h'], weights['decoder_h'], biases['encoder_b'], biases['decoder_b']]))
        return loss

    # Training loop
    prev_loss = 0
    for k in range(1, FLAGS.iterations + 1):
        loss = train_step(X, mask)
        lpentry = loss / cells
        change = abs(prev_loss - lpentry)
        if change <= FLAGS.threshold:
            print("Reached the threshold value.")
            break
        prev_loss = lpentry
        if FLAGS.debug:
            if (k - 1) % FLAGS.debug_display_step == 0:
                print(f'Step {k} : Total loss: {loss:.6f}, Loss per Cell : {lpentry:.6f}, Change : {change:.6f}')
                with open(FLAGS.log_file, 'a') as log:
                    log.write(f'{k}\t{loss:.6f}\t{lpentry:.6f}\t{change:.6f}\n')

    # Saving the final imputed matrix
    imputed_count_matrix = decoder(encoder(X)).numpy()
    scipy.io.savemat(FLAGS.imputed_save + ".mat", mdict={"arr": imputed_count_matrix})

    finish_time = datetime.datetime.now()
    print(f"[info] Total time taken = {finish_time - start_time}")
