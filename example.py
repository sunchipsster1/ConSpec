# Example of how to use ConSpec in a training loop:

env = create_env()
algo = get_algo()
conspec = ConSpec(state_shape=env.observation_space.shape)

num_episodes = 100000
for ep in num_episodes:
    obs = env.reset()
    done = False
    trajectory = []
    while not done:
        action = algo.act(obs)
        next_obs, extrinsic_reward, done, _ = env.step(action)

        transition = (obs, action, next_obs, extrinsic_reward)
        trajectory.append(transition)
        obs = next_obs
        conspec.train_prototypes_step()
        if done:
            break
    conspec.add_trajectory(trajectory)
    conspec.train_prototypes_step()
    algo.train(trajectory)

    if ep % 1000 == 0:
        # Could be useful to get the prototypes for logging
        prototypes = conspec.get_prototypes()
        # Could be useful to get the intrinsic reward for logging
        intrinsic_reward = conspec.get_intrinsic_reward_for_trajectory(trajectory)
        # Could be useful to get the intrinsic reward for logging
        intrinsic_reward = conspec.get_intrinsic_reward_for_transition(transition)
        # Could be useful to save the prototypes for logging
        save_prototypes(prototypes)
        # Could be useful to save the intrinsic reward for logging
        save_intrinsic_reward(intrinsic_reward)