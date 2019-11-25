import gym

class EnvSampler():
    def __init__(self, env, agent, max_path_length=1000):
        self.env = env
        self.agent = agent

        self.path_length = 0
        self.current_state = None
        self._max_path_length = max_path_length

    def sample(self):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        action = self.agent.select_action(self.current_state, eval=True)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
