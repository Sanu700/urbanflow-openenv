from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import TrafficEnv

app = FastAPI()
env = TrafficEnv()

class ActionRequest(BaseModel):
    action: str | None = None

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs, "reward": 0.0, "done": False, "info": {}}

@app.post("/step")
def step(req: ActionRequest):
    obs, reward, done, info = env.step(req.action)
    return {"observation": obs, "reward": float(reward), "done": bool(done), "info": info}

@app.get("/state")
def state():
    return env.state()