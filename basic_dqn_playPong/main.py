#!/usr/bin/env python3
from lib import wrappers
from lib import dqn_model
from lib.my_utils import *

env = wrappers.make_env(DEFAULT_ENV_NAME)
dqns = dqn_model.DQN_keras(env.observation_space.shape, env.action_space.n)
buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
total_rewards = []
frame_idx = 0

while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    reward = agent.play_step(dqns.model, epsilon)
    if reward is not None:
        total_rewards.append(reward)
        print("%d frames %d games, reward % .3f, eps %.2f, mean reward %.3f" % (
        frame_idx, len(total_rewards), reward, epsilon, np.mean(total_rewards[-100:])))

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        dqns.tgt_model.set_weights(dqns.model.get_weights())

    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, dones, next_states = batch
    label = calc_label(batch, dqns.tgt_model)
    q_eval = dqns.model.predict(states)
    batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
    q_target = q_eval.copy()
    q_target[batch_index, actions] = label
    dqns.model.fit(states, q_target, epochs=1, verbose=0)
