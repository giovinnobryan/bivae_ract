import itertools as it

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

class CriticModelMixin:
    """
    This provides the setup for having a model with a Critic

    What's the signature? Pretty much, I want to replace phase 3 with minimizing WARP.
    There are some of these that no longer make sense, but that's okay. For example,
    logging expected-ndcg or whatever is no longer a sensible metric.

    ac_train_op
    critic_train_op
    critic_error (maybe)

    I think that's roughly it. So, critic_train_op should be a no-op. There is a tf.no_op apparently.
    And ac_train_op should just be minimizing WARP. 

    

    """

    def construct_placeholders(self):
        """
        Any time I need a placeholder, it'll be here. And everything will inherit from this guy.
        That's probably the cleanest way of doing this.

        Sets fields:
            keep_prob_ph:
                The placeholder that determines the dropout fraction. During training, it should
                be something like 0.5. During validation, it should always be 1.
            self.epoch:
                This is useful for doing things that need to scale based on epoch. For example, the
                kl_scaler function will go here.
            self.stddev_effect_on_latent_dim_scaler:
                It's unfortunate that I need this one, but when we're doing testing, we want to
                sample the mean, not anything else.
            self.train_batch_norm:
                I'm not really sure about this one.. But I think it determines if you're training the
                batch-norm or not...

        """
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[])
        self.kl_loss_scaler = tf.placeholder_with_default(0.0, shape=None)
        self.stddev_effect_on_latent_dim_scaler = tf.placeholder_with_default(0., shape=[])
        self.train_batch_norm = tf.placeholder_with_default(False, shape=[])
        self.epoch = tf.placeholder_with_default(0.0, shape=[])

    def forward_pass_critic(self):
        with tf.variable_scope("CriticInputVector"):
            self._create_critic_input_vector()

        self.batch_norm_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope=".*CriticInputVector.*")

        with tf.variable_scope("RestOfCritic"):
            self._create_critic_network()

        print("number of batch_norm_update_ops: {}".format(len(self.batch_norm_update_ops)))

    def _create_critic_input_vector(self):
        critic_inputs = []
        if not getattr(self, 'omit_num_seen_from_critic', False):
            print("NOT OMITTING NUM SEEN")
            critic_inputs.append(self.number_of_seen_items)
        else:
            print("OMITTING NUM SEEN")
        if not getattr(self, 'omit_num_unseen_from_critic', False):
            print('NOT OMITTING NUM UNSEEN')
            critic_inputs.append(self.number_of_unseen_items)
        else:
            print("OMITTING NUM UNSEEN")

        print("Always doing actor error, of course.")
        critic_inputs.append(self.actor_error)

        unnormalized_ac_input = tf.stack(critic_inputs, axis=-1)

        self.ac_input = tf.contrib.layers.batch_norm(
            unnormalized_ac_input,
            is_training=self.train_batch_norm,
            trainable=False,  #Don't know what scale is really, but it says if relu next, don't use.
            renorm=True,  #Not sure. But makes it use closer stats for training and testing.
        )

    def _get_final_activation_fn(self):
        unbounded_metrics = ['DCG']
        if getattr(self, 'evaluation_metric') in unbounded_metrics:
            print("\n\nwe want it to be positive, so we'll use softplus. I think that's fair.\n\n")
            return tf.nn.softplus
        else:
            return tf.nn.sigmoid

    def _create_critic_network(self):

        # THIS IS A REALLY IMPORTANT LINE. MAKES THE OUTPUT UNBOUNDED FOR DCG...
        critic_activation_fn = self._get_final_activation_fn()

        h = slim.fully_connected(self.ac_input, 100, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, 10, activation_fn=tf.nn.relu)
        critic_output = slim.fully_connected(h, 1, activation_fn=critic_activation_fn)
        critic_output = tf.squeeze(critic_output)
        self.critic_output = critic_output

    def _return_unnormalized_dcg_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        # This is NDCG, but not normalized by IDCG...
        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        # The False at the end is what makes it UNNORMALIZED
        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 100, input_batch, False],
                          tf.float32)

    def _return_ndcg_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        In our case, true_outputs should be the remaining_input field.
        Our_outputs is the softmax output.
        input_batch is the ones that you wanna zero. That's important, because otherwise you see the
        predictions from the ones you knew about.

        But, it's still weird that it messed it up so grandly... not sure what happened there.
        """
        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 100, input_batch],
                          tf.float32)

    def _return_ap_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        In our case, true_outputs should be the remaining_input field.
        Our_outputs is the softmax output.
        input_batch is the ones that you wanna zero. That's important, because otherwise you see the
        predictions from the ones you knew about.

        But, it's still weird that it messed it up so grandly... not sure what happened there.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(average_precision_at_k_batch, [our_outputs, true_outputs, 100, input_batch],
                          tf.float32)

    def _return_recall_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(Recall_at_k_batch, [our_outputs, true_outputs, 100, input_batch],
                          tf.float32)

    def _return_ndcg_at_200_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 200, input_batch],
                          tf.float32)


    def _return_ndcg_at_50_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 50, input_batch],
                          tf.float32)

    def _return_ndcg_at_20_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 20, input_batch],
                          tf.float32)

    def _return_ndcg_at_5_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 5, input_batch],
                          tf.float32)

    def _return_ndcg_at_3_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 3, input_batch],
                          tf.float32)

    def _return_ndcg_at_1_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
        """
        I'm doing recall@100, because it would be hard to justify having this one be the only outlier.
        """

        assert our_outputs is not None
        assert true_outputs is not None
        assert input_batch is not None

        return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 1, input_batch],
                          tf.float32)


    def _build_critic_reg(self):
        with tf.variable_scope("CriticRegularization"):
            print('changed to 1e-4 from 1e-6, just so I know.'
                 )  # I think keeping it regularized might help...
            reg = l2_regularizer(
                1e-4
            )  #just to start I'll hard code it, see how big it is. compared to error. Keep it conservative to start.
            reg_var = apply_regularization(reg, self.critic_forward_variables)
            reg_var = 2 * reg_var
            reg_var = reg_var / self.batch_size  # So that it scales with batch_size just like other errors.
        self.critic_regularization_loss = reg_var

    def construct_critic_error(self):
        true_dcg = self._return_unnormalized_dcg_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        true_ndcg = self._return_ndcg_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        true_ndcg_at_200 = self._return_ndcg_at_200_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        true_ndcg_at_5 = self._return_ndcg_at_5_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        true_ndcg_at_3 = self._return_ndcg_at_3_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        true_ndcg_at_1 = self._return_ndcg_at_1_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        # true_ap = self._return_ap_given_args(
        #     our_outputs=self.prediction,
        #     true_outputs=self.remaining_input,
        #     input_batch=self.network_input)
        
        true_recall = self._return_recall_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input,
            input_batch=self.network_input)

        if self.evaluation_metric == 'NDCG':
            print("Evaluating with NDCG")
            evaluation_metric = true_ndcg

        elif self.evaluation_metric == 'NDCG_AT_200':
            evaluation_metric = true_ndcg_at_200
        elif self.evaluation_metric == 'NDCG_AT_5':
            evaluation_metric = true_ndcg_at_5
        elif self.evaluation_metric == 'NDCG_AT_3':
            evaluation_metric = true_ndcg_at_3
        elif self.evaluation_metric == 'NDCG_AT_1':
            evaluation_metric = true_ndcg_at_1

        elif self.evaluation_metric == 'DCG':
            evaluation_metric = true_dcg

        # elif self.evaluation_metric == 'AP':
        #     print("Evaluating with AP")
        #     evaluation_metric = true_ap
        # elif self.evaluation_metric == 'RECALL':
        #     evaluation_metric = true_recall
        # elif self.evaluation_metric == 'NDCG_AT_200':
        #     evaluation_metric = true_ndcg_at_200
        # elif self.evaluation_metric == 'NDCG_AT_50':
        #     evaluation_metric = true_ndcg_at_50
        # elif self.evaluation_metric == 'NDCG_AT_20':
        #     evaluation_metric = true_ndcg_at_20
        # elif self.evaluation_metric == 'NDCG_AT_5':
        #     evaluation_metric = true_ndcg_at_5
        else:
            raise ValueError("evaluation_metric must be one of NDCG, AP, RECALL, or one of the NDCG ones. Instead got {}".format(
                self.evaluation_metric))

        critic_error = (evaluation_metric - self.critic_output)**2
        self._build_critic_reg()
        mean_critic_error = tf.reduce_mean(critic_error) + self.critic_regularization_loss

        self.true_dcg = true_dcg
        self.true_ndcg = true_ndcg
        # self.true_ap = true_ap
        self.true_recall = true_recall

        self.true_ndcg_at_200 = true_ndcg_at_200
        # self.true_ndcg_at_50 = true_ndcg_at_50
        # self.true_ndcg_at_20 = true_ndcg_at_20
        self.true_ndcg_at_5 = true_ndcg_at_5
        self.true_ndcg_at_3 = true_ndcg_at_3
        self.true_ndcg_at_1 = true_ndcg_at_1

        self.critic_error = critic_error
        self.mean_critic_error = mean_critic_error
        self.true_evaluation_metric = evaluation_metric

    # Just the default implementation.
    # def construct_critic_training(self):
    #     pass

    def construct_ac_error(self):
        self.ac_error = -1 * self.critic_output
        self.mean_ac_error = tf.reduce_mean(
            self.ac_error) + (self.ac_reg_loss_scaler * self.actor_regularization_loss)

    # Just the default implementation.
    # def construct_ac_training(self):
    #     pass

    def construct_actor_training(self):
        print("number of BN update ops: {}".format(len(self.batch_norm_update_ops)))
        with tf.control_dependencies(self.batch_norm_update_ops):
            train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(
                self.mean_actor_error, var_list=self.actor_forward_variables)  #TODO: var_list part

        self.actor_train_op = train_op

    def create_validation_ops(self):
        # This is going to be a sexy graph replace.
        # network-input is the same. The actor-error-mask is different, although honestly, it doesn't matter
        # what actor-error is for validation.
        vad_true_ndcg, vad_critic_error, vad_prediction = \
            tf.contrib.graph_editor.graph_replace(
                [self.true_ndcg, self.critic_error, self.prediction],
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
                })

        # vad_true_ndcg, vad_true_ap, vad_true_recall, vad_critic_error, vad_prediction = \
        #     tf.contrib.graph_editor.graph_replace(
        #         [self.true_ndcg, self.true_ap, self.true_recall, self.critic_error, self.prediction],
        #         {
        #             self.remaining_input : self.heldout_batch,
        #             self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
        #         })

        # vad_true_ndcg_at_200, vad_true_ndcg_at_50, vad_true_ndcg_at_20, vad_true_ndcg_at_5 = \
        #     tf.contrib.graph_editor.graph_replace(
        #         [self.true_ndcg_at_200, self.true_ndcg_at_50, self.true_ndcg_at_20, self.true_ndcg_at_5],
        #         {
        #             self.remaining_input : self.heldout_batch,
        #             self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
        #         })

        vad_actor_error, vad_critic_output = \
            tf.contrib.graph_editor.graph_replace(
                [self.actor_error, self.critic_output],
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
                })

        vad_true_evaluation_metric = \
            tf.contrib.graph_editor.graph_replace(
                self.true_evaluation_metric,
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
                })
        
        self.vad_true_evaluation_metric = vad_true_evaluation_metric

        self.vad_true_ndcg = vad_true_ndcg
        # self.vad_true_ap = vad_true_ap
        # self.vad_true_recall = vad_true_recall

        # self.vad_true_ndcg_at_200 = vad_true_ndcg_at_200
        # self.vad_true_ndcg_at_50 = vad_true_ndcg_at_50
        # self.vad_true_ndcg_at_20 = vad_true_ndcg_at_20
        # self.vad_true_ndcg_at_5 = vad_true_ndcg_at_5
        

        self.vad_critic_error = vad_critic_error
        self.vad_prediction = vad_prediction
        self.vad_actor_error = vad_actor_error
        self.vad_critic_output = vad_critic_output


    def create_logging_ops(self):
        # tf.summary.scalar('mean_actor_error', self.mean_actor_error)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)