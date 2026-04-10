import random

class TrafficEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.step_count = 0
        self.max_steps = {"easy": 3, "medium": 4, "hard": 5}[task]

    def reset(self):
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "traffic": [random.randint(1, 10) for _ in range(4)],
            "task": self.task
        }

    def step(self, action):
        self.step_count += 1

        # Simple reward logic
        traffic = sum([random.randint(1, 10) for _ in range(4)])
        reward = max(0.0, 1.0 - traffic / 40)

        done = self.step_count >= self.max_steps

        return self._get_obs(), reward, done, {"error": "null"}

    def state(self):
        return {"step": self.step_count}