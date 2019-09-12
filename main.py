# MIT License, see LICENSE

import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf

from absl import app as absl_app
from absl import flags

from models.model import get_model, get_loss
from utils.data_reader import DataReader
from utils.view_steering_model import render_steering_tf

from utils.flags import core as flags_core
from utils.logs import hooks_helper
from utils.misc import distribution_utils

FLAGS = flags.FLAGS

def parse_args():
    """Parse arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Trains a self-steering car model in single-instance or distributed mode.
                            For distributed mode, the script will use few environment variables as defaults:
                            JOB_NAME, TASK_INDEX, PS_HOSTS, and WORKER_HOSTS. These environment variables will be
                            available on distributed Tensorflow jobs on Clusterone platform by default.
                            If running this locally, you will need to set these environment variables
                            or pass them in as arguments (i.e. python main.py --job_name worker --task_index 0
                            --worker_hosts "localhost:2222,localhost:2223" --ps_hosts "localhost:2224").
                            If these are not set, the script will run in non-distributed (single instance) mode.''')

    # Configuration for distributed task
    parser.add_argument('--job_name', type=str, default=os.environ.get('JOB_NAME', None), choices=['worker', 'ps'],
                        help='Task type for the node in the distributed cluster. Worker-0 will be set as master.')
    parser.add_argument('--task_index', type=int, default=os.environ.get('TASK_INDEX', 0),
                        help='Worker task index, should be >= 0. task_index=0 is the chief worker.')
    parser.add_argument('--ps_hosts', type=str, default=os.environ.get('PS_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--worker_hosts', type=str, default=os.environ.get('WORKER_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')

    # Experiment related parameters
    parser.add_argument('--absolute_data_path', type=str, default='/datasets/self-driving-demo-data/',
                        help='Using this will ignore other data path arguments.')

    # Model params
    parser.add_argument('--dropout_rate1', type=float, default=0.2,
                        help='Dropout rate after the convolutional layers.')
    parser.add_argument('--dropout_rate2', type=float, default=0.5,
                        help='Dropout rate after the dense layer.')
    parser.add_argument('--fc_dim', type=int, default=512,
                        help='Number of dimensions in the dense layer.')
    parser.add_argument('--nogood', action='store_true',
                        help='Ignore "goods" filters')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate used in Adam optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.0001,
                        help='Exponential decay rate of the learning rate per step.')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to use during training and evaluation.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Max number of steps to train for.')
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'],
                        help='TF logging level. To log intermediate results, set this to INFO or DEBUG.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use to prepare data')
    parser.add_argument('--max_ckpts', type=int, default=2,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--ckpt_steps', type=int, default=100,
                        help='How frequently to save a model checkpoint')
    parser.add_argument('--save_summary_steps', type=int, default=10,
                        help='How frequently to save TensorBoard summaries')
    parser.add_argument('--log_step_count_steps', type=int, default=10,
                        help='How frequently to log loss & global steps/s')
    parser.add_argument('--eval_secs', type=int, default=60,
                        help='How frequently to run evaluation step. ' +
                             'By default, there is no evaluation dataset, thus effectively no evaluation.')

    # Parse args
    opts = parser.parse_args()

    
    opts.train_data = os.path.join(opts.absolute_data_path, 'camera/training/*.h5')

    opts.model_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/models') + '/self-driving')
    opts.ps_hosts = opts.ps_hosts.split(',') if opts.ps_hosts else []
    opts.worker_hosts = opts.worker_hosts.split(',') if opts.worker_hosts else []

    return opts

def define_self_driving_flags():

    flags.DEFINE_integer('eval_secs', os.environ.get('EVAL_SECS', 600), 'How frequently to run evaluation step')
    flags.DEFINE_integer('ckpt_steps', os.environ.get('CKPT_STEPS', 600), 'How frequently to save a model checkpoin')
    flags.DEFINE_integer('max_ckpts', 5, 'Maximum number of checkpoints to keep')
    flags.DEFINE_integer('max_steps', os.environ.get('MAX_STEPS', 150000), 'Max steps')
    flags.DEFINE_integer('save_summary_steps', 100, 'How frequently to save TensorBoard summaries')
    flags.DEFINE_integer('log_step_count_steps', 100, 'How frequently to log loss & global steps/s')

    flags.DEFINE_integer('num_threads', 1, 'Number of threads to use to prepare data')
    # Model params
    flags.DEFINE_float('dropout_rate1', 0.2,
                        'Dropout rate after the convolutional layers.')
    flags.DEFINE_float('dropout_rate2', 0.5,
                        'Dropout rate after the dense layer.')
    flags.DEFINE_integer('fc_dim', 512,
                        'Number of dimensions in the dense layer.')
    
    flags.DEFINE_float('learning_rate', 0.0001,
                        'Initial learning rate used in Adam optimizer.')
    flags.DEFINE_float('learning_decay', 0.0001,
                        'Exponential decay rate of the learning rate per step.')

    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    data_dir = '/datasets/self-driving-demo-data/camera/training/*.h5'
    model_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/models') + '/self-driving')
    export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/models'))
    flags.adopt_module_key_flags(flags_core)
    flags_core.set_defaults(data_dir=data_dir,
                            model_dir=model_dir,
                            export_dir=export_dir,
                            train_epochs=int(os.environ.get('TRAIN_EPOCHS', 40)),
                            epochs_between_evals=int(os.environ.get('EPOCHS_EVAL', 100)),
                            batch_size=int(os.environ.get('BATCH_SIZE', 100)),
                            )
                            
def make_tf_config(opts):
    """Returns TF_CONFIG that can be used to set the environment variable necessary for distributed training"""
    try:
        import gradient_sdk
        gradient_sdk.get_tf_config()

        job_name = os.environ['TYPE']
        if job_name == 'master':
            job_name = 'worker'
        task_index = int(os.environ['INDEX'])
        ps_hosts = gradient_sdk.ps_hosts().split(',')
        worker_hosts = gradient_sdk.worker_hosts().split(',')
        if job_name == 'ps':
            ps_hosts[task_index] = 'localhost:%s'%(ps_hosts[task_index].split(':')[-1])
        elif job_name == 'worker':
            worker_hosts[task_index] = 'localhost:%s'%(worker_hosts[task_index].split(':')[-1])
        return os.getenv('TF_CONFIG')

    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None
        return {}

    # tf_config = {
    #     'task': {
    #         'type': opts.job_name,
    #         'index': opts.task_index
    #     },
    #     'cluster': {
    #         'master': [opts.worker_hosts[0]],
    #         'worker': opts.worker_hosts,
    #         'ps': opts.ps_hosts
    #     },
    #     'environment': 'cloud'
    # }

    # # Nodes may need to refer to itself as localhost
    # local_ip = 'localhost:' + tf_config['cluster'][opts.job_name][opts.task_index].split(':')[1]
    # tf_config['cluster'][opts.job_name][opts.task_index] = local_ip
    # if opts.job_name == 'worker' and opts.task_index == 0:
    #     tf_config['task']['type'] = 'master'
    #     tf_config['cluster']['master'][0] = local_ip
    # return tf_config


def read_row(filenames):
    """Read a row of data from list of H5 files"""
    reader = DataReader(filenames)
    x, y, s = reader.read_row_tf()
    x.set_shape((3, 160, 320))
    y.set_shape(1)
    s.set_shape(1)
    return x, y, s


def get_input_fn(files, opts, is_train=True):
    """Returns input_fn.  is_train=True shuffles and repeats data indefinitely"""
    def input_fn():
        with tf.device('/cpu:0'):
            x, y, s = read_row(files)
            if is_train:
                X, Y, S = tf.train.shuffle_batch([x, y, s],
                                                 batch_size=opts.batch_size,
                                                 capacity=5 * opts.batch_size,
                                                 min_after_dequeue=2 * opts.batch_size,
                                                 num_threads=opts.num_threads)
            else:
                X, Y, S = tf.train.batch([x, y, s],
                                         batch_size=opts.batch_size,
                                         capacity=5 * opts.batch_size,
                                         num_threads=opts.num_threads)
            return {'features': X, 's': S}, Y
    return input_fn


def get_model_fn(opts):
    """Return model fn to be used for Estimator class"""
    def model_fn(features, labels, mode):
        features, s = features['features'], features['s']
        y_pred = get_model(features, opts)

        tf.summary.image("green-is-predicted", render_steering_tf(features, labels, s, y_pred))

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prediction': y_pred}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = get_loss(y_pred, labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            lr = tf.train.exponential_decay(learning_rate=opts.learning_rate,
                                            global_step=global_step,
                                            decay_steps=1,
                                            decay_rate=opts.learning_decay)
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    return model_fn


def main():
    """Main"""
    opts = flags.FLAGS
    # Create an estimator
    model_function = get_model_fn(opts)

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=opts.inter_op_parallelism_threads,
        intra_op_parallelism_threads=opts.intra_op_parallelism_threads,
        allow_soft_placement=True)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        flags_core.get_num_gpus(opts), opts.all_reduce_alg)


    config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_steps=opts.ckpt_steps,
        keep_checkpoint_max=opts.max_ckpts,
        log_step_count_steps=opts.log_step_count_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=opts.model_dir,
        config=config)

    # Create input fn
    # We do not provide evaluation data, so we'll just use training data for both train & evaluation.
    train_input_fn = get_input_fn(opts.train_data, opts, is_train=True)
    eval_input_fn = get_input_fn(opts.train_data, opts, is_train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=opts.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=1,
                                      start_delay_secs=0,
                                      throttle_secs=opts.eval_secs)

    # Train and evaluate!
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
     # Export the model if node is master and export_dir is set and if experiment is multinode - check if its master
    if os.environ.get('PS_CONFIG') and os.environ.get('TYPE') != 'master':
        tf.logging.debug('No model was exported')
        return

    if opts.model_dir:
        tf.logging.debug('Starting to Export model to {}'.format(str(opts.model_dir)))
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        estimator.export_savedmodel(opts.model_dir, input_fn,
                                           strip_default_attrs=True)
        tf.logging.debug('Model Exported')

if __name__ == "__main__":
    args = parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    define_self_driving_flags()

    tf.logging.debug('=' * 20 + ' Environment Variables ' + '=' * 20)
    for k, v in os.environ.items():
        tf.logging.debug('{}: {}'.format(k, v))

    tf.logging.debug('=' * 20 + ' Arguments ' + '=' * 20)
    for k, v in sorted(args.__dict__.items()):
        if v is not None:
            tf.logging.debug('{}: {}'.format(k, v))
    

    #TF_CONFIG = make_tf_config(args)
    import gradient_sdk
    gradient_sdk.get_tf_config()
    tf.logging.debug('=' * 20 + ' TF_CONFIG ' + '=' * 20)
    tf.logging.debug(os.environ.get('TF_CONFIG'))
    #os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

    tf.logging.info('=' * 20 + ' Train starting ' + '=' * 20)
    absl_app.run(main)