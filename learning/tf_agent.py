import numpy as np
#import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf; print(tf.__version__); tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from abc import abstractmethod

from learning.rl_agent import RLAgent
from util.logger import Logger
from learning.tf_normalizer import TFNormalizer


# Create a custom layer for part of the model
class AgentLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):

        argi = 0
        for arg in args:
            if (argi == 0):
                self.sess = arg
            if (argi == 1):
                self.get_state_size = arg
            if (argi == 2):
                self.get_goal_size = arg
            if (argi == 3):
                self.get_action_size = arg

            Logger.print("argi: " + str(argi))
            argi = argi + 1

        super(AgentLayer, self).__init__(*args, **kwargs)


    def build(self, input_shape):
        self.w = self.add_weight(
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.keras.initializers.ones(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)

        with tf.device('cpu:0'):
            with self.sess.as_default(): #, self.graph.as_default():
                # with scope agent
                    # with scope resource
                #with tf.variable_scope(self.RESOURCE_SCOPE):
                    #with tf.variable_scope(self.RESOURCE_SCOPE):
                self.s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(), self.world.env.build_state_norm_groups(self.id))
                self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id),
                                         1 / self.world.env.build_state_scale(self.id))

                self.g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
                self.g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id),
                                         1 / self.world.env.build_goal_scale(self.id))

                self.a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
                self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id),
                                         1 / self.world.env.build_action_scale(self.id))

    # Call method will sometimes get used in graph mode,
    # training will get turned into a tensor
    @tf.function
    def call(self, inputs, training=None):
        if training:
            return inputs + self.w
        else:
            return inputs + self.w * 0.5


class TFAgent(RLAgent):
    RESOURCE_SCOPE = 'resource'
    SOLVER_SCOPE = 'solvers'

    def __init__(self, world, id, json_data):
        self.tf_scope = 'agent'
        #self.graph = tf.Graph()
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
 #       gpu_options = tf.GPUOptions(allow_growth = True)
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth=True)


        #config = tf.ConfigProto(gpu_options = gpu_options)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        #config.gpu_options.allow_growth = True

        #import tensorflow as tf
        #tf.config.gpu.set_per_process_memory_fraction(0.75)
        #tf.config.gpu.set_per_process_memory_growth(True)
        #session = InteractiveSession(config=config)
        #self.sess = tf.Session(config=config,graph=self.graph)


        self.init_session_mem()

        self.agent_layer = AgentLayer(self.sess, self.get_state_size(), self.get_goal_size(), self.get_action_size())
        #Logger.print('agent_layer([1]).numpy(): ' + self.agent_layer([1]).numpy())

        self.json_data = json_data
        super().__init__(world, id, json_data)
        self._build_graph(json_data)
        self._init_normalizers()
        return

    def __del__(self):
        with tf.device('cpu:0'):
            self.sess.close()
        return

    def init_session_mem(self):
        Logger.print('[TFAgent] Init session -> tf.Graph(), tf.Session(...) called.')

        with tf.device('cpu:0'):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth=True)
            config = tf.ConfigProto(gpu_options = gpu_options)
            self.graph = tf.Graph()
            self.sess = tf.Session(config=config,graph=self.graph)
        return


    def clear_session_mem(self):

        with tf.device('cpu:0'):

            Logger.print('[TFAgent] Clear session -> tf.Graph(), tf.Session(...) called.')
            self.sess.close()

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth=True)
            config = tf.ConfigProto(gpu_options = gpu_options)
            self.graph = tf.Graph()
            self.sess = tf.Session(config=config,graph=self.graph)
        return

    def save_model(self, out_path):
        with tf.device('cpu:0'):
            with self.sess.as_default(), self.graph.as_default():
                try:
                    save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
                    Logger.print('Model saved to: ' + save_path)
                except:
                    Logger.print("Failed to save model to: " + save_path)
        return


    def load_model(self, in_path):
        with tf.device('cpu:0'):
            with self.sess.as_default(), self.graph.as_default():
                Logger.print('Restoring checkpoint for model: ' + in_path)
                self.saver.restore(self.sess, in_path)
                self._load_normalizers()
                Logger.print('Model loaded from: ' + in_path)
        return

    def _get_output_path(self):
        assert(self.output_dir != '')
        file_path = self.output_dir + '/agent' + str(self.id) + '_model.ckpt'
        return file_path

    def _get_int_output_path(self):
        assert(self.int_output_dir != '')
        file_path = self.int_output_dir + ('/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
        return file_path

    def _build_graph(self, json_data):
        with tf.device('cpu:0'):

            with self.sess.as_default(), self.graph.as_default():
                with tf.variable_scope(self.tf_scope):
                    self._build_nets(json_data)

                    with tf.variable_scope(self.SOLVER_SCOPE):
                        self._build_losses(json_data)
                        self._build_solvers(json_data)

                    self._initialize_vars()
                    self._build_saver()
        return

    def _init_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            # update normalizers to sync the tensorflow tensors
            self.s_norm.update()
            self.g_norm.update()
            self.a_norm.update()
        return

    @abstractmethod
    def _build_nets(self, json_data):
        pass

    @abstractmethod
    def _build_losses(self, json_data):
        pass

    @abstractmethod
    def _build_solvers(self, json_data):
        pass

    def _tf_vars(self, scope=''):
        with tf.device('cpu:0'):
            with self.sess.as_default(), self.graph.as_default():
                res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + scope)
                assert len(res) > 0
                return res

    def _build_normalizers(self):
        Logger.print('[TFAgent] Build normalizers -> TFNormalizer (s_norm, g_norm, a_norm) called.')

        with tf.device('cpu:0'):

            with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
                with tf.variable_scope(self.RESOURCE_SCOPE):
                #with tf.variable_scope(self.RESOURCE_SCOPE):
                    self.s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(), self.world.env.build_state_norm_groups(self.id))
                    self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id),
                                             1 / self.world.env.build_state_scale(self.id))

                    self.g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
                    self.g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id),
                                             1 / self.world.env.build_goal_scale(self.id))

                    self.a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
                    self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id),
                                             1 / self.world.env.build_action_scale(self.id))
        return

    def _load_normalizers(self):
        self.s_norm.load()
        self.g_norm.load()
        self.a_norm.load()
        return

    def _update_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            super()._update_normalizers()
        return

    def _initialize_vars(self):
        with tf.device('cpu:0'):
            self.sess.run(tf.global_variables_initializer())
        return

    def _build_saver(self):
        with tf.device('cpu:0'):

            vars = self._get_saver_vars()
            self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with tf.device('cpu:0'):

            with self.sess.as_default(), self.graph.as_default():
                vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
                vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
                #vars = [v for v in vars if '/target/' not in v.name]
                assert len(vars) > 0
        return vars
    
    def _weight_decay_loss(self, scope):
        with tf.device('cpu:0'):
            vars = self._tf_vars(scope)
            vars_no_bias = [v for v in vars if 'bias' not in v.name]
            loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
            return loss

    def _train(self):
        with tf.device('cpu:0'):
            with self.sess.as_default(), self.graph.as_default():
                    super()._train()
        return