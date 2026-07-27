"""Compatibility adapters for legacy training code."""


class LegacyDoneAdapter:
    """Converts Gymnasium step output to the old four-value Gym contract."""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class LegacyTrainingAdapter:
    """Matches the return contract used by the existing MAPPO/Ego/HDV code."""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        seed = kwargs.get("testing_seeds", kwargs.get("seed"))
        options = {key: value for key, value in kwargs.items() if key not in {"seed", "testing_seeds"}}
        obs, _ = self.env.reset(seed=seed, options=options or None)
        action_mask = None
        return obs, action_mask, self.env.obs2, self.env.obs3

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info, self.env.obs2, self.env.obs3

    def __getattr__(self, name):
        return getattr(self.env, name)
