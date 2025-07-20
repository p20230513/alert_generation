import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from collections import deque
import random

class ReplayBuffer:
    """A simple replay buffer for storing experiences."""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class SacApAgent:
    """Soft Actor-Critic Agent for Alert Prioritization."""
    def __init__(self, state_dim, action_dim, action_bound, buffer_size=100000,
                 lr_actor=0.001, lr_critic=0.002, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound  # For scaling actions

        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft target update factor
        self.alpha = alpha  # Entropy regularization coefficient

        # Actor-Critic Networks
        self.actor = self._build_actor()
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()

        # Target Networks
        self.target_actor = self._build_actor()
        self.target_critic_1 = self._build_critic()
        self.target_critic_2 = self._build_critic()

        # Initialize target networks with the same weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        # Optimizers
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_1_optimizer = Adam(learning_rate=lr_critic)
        self.critic_2_optimizer = Adam(learning_rate=lr_critic)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _build_actor(self):
        """Builds the actor network that outputs a distribution for actions."""
        state_input = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(state_input)
        x = Dense(256, activation='relu')(x)
        # Output mean and log_std for a Gaussian distribution
        mean = Dense(self.action_dim, activation='tanh')(x)
        log_std = Dense(self.action_dim, activation='softplus')(x)

        # Scale mean to action range
        mean = mean * self.action_bound

        model = Model(state_input, [mean, log_std])
        return model

    def _build_critic(self):
        """Builds the critic network that evaluates state-action pairs."""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        concat = Concatenate()([state_input, action_input])

        x = Dense(256, activation='relu')(concat)
        x = Dense(256, activation='relu')(x)
        q_value = Dense(1)(x)

        model = Model([state_input, action_input], q_value)
        return model

    def get_action(self, state):
        """Get a stochastic action from the actor network."""
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        mean, log_std = self.actor(state)
        
        # Create a Normal distribution and sample an action
        std = tf.exp(log_std)
        normal_dist = tfp.distributions.Normal(mean, std)
        raw_action = normal_dist.sample()
        
        # Apply tanh to squash the action between -1 and 1
        squashed_action = tf.tanh(raw_action)
        
        # The output action represents priorities. We can scale it if needed.
        # For now, let's return the squashed action and its log probability.
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= tf.math.log(1.0 - tf.square(squashed_action) + 1e-6)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)

        return squashed_action[0].numpy(), log_prob[0].numpy()

    def train(self, batch_size):
        """Train the agent by sampling from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return  # Not enough samples to train

        minibatch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.reshape(rewards, (batch_size, 1))
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        dones = tf.reshape(dones, (batch_size, 1))

        # --- Train Critic Networks ---
        with tf.GradientTape(persistent=True) as tape:
            # Get next actions and their log probabilities from the target actor
            next_mean, next_log_std = self.target_actor(next_states)
            next_std = tf.exp(next_log_std)
            next_dist = tfp.distributions.Normal(next_mean, next_std)
            next_raw_actions = next_dist.sample()
            next_actions = tf.tanh(next_raw_actions)

            next_log_probs = next_dist.log_prob(next_raw_actions)
            next_log_probs -= tf.math.log(1.0 - tf.square(next_actions) + 1e-6)
            next_log_probs = tf.reduce_sum(next_log_probs, axis=1, keepdims=True)

            # Get target Q-values from target critics
            target_q1 = self.target_critic_1([next_states, next_actions])
            target_q2 = self.target_critic_2([next_states, next_actions])
            target_q = tf.minimum(target_q1, target_q2)

            # Add entropy term to the target Q-value
            soft_q_target = rewards + (1.0 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)

            # Get current Q-values for calculating loss
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            
            critic_1_loss = tf.reduce_mean(tf.square(current_q1 - soft_q_target))
            critic_2_loss = tf.reduce_mean(tf.square(current_q2 - soft_q_target))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

        # --- Train Actor Network ---
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            raw_actions = dist.sample()
            actions_pred = tf.tanh(raw_actions)
            
            log_probs = dist.log_prob(raw_actions)
            log_probs -= tf.math.log(1.0 - tf.square(actions_pred) + 1e-6)
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            
            q1_pred = self.critic_1([states, actions_pred])
            q2_pred = self.critic_2([states, actions_pred])
            q_pred = tf.minimum(q1_pred, q2_pred)
            
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_pred)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        self._update_target_networks()

    def _update_target_networks(self):
        """Perform soft updates on target networks."""
        def soft_update(target_weights, weights, tau):
            for (a, b) in zip(target_weights, weights):
                a.assign(b * tau + a * (1 - tau))

        soft_update(self.target_actor.variables, self.actor.variables, self.tau)
        soft_update(self.target_critic_1.variables, self.critic_1.variables, self.tau)
        soft_update(self.target_critic_2.variables, self.critic_2.variables, self.tau)