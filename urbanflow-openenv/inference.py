import os
from openai import OpenAI
from env.environment import TrafficEnv

# Env vars
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_action(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"Traffic: {obs}. Give signal_config like [0,1,0,1]"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "null"


def run_task(task):
    env = TrafficEnv(task=task)
    obs = env.reset()

    step_count = 0
    rewards = []
    success = False

    print(f"[START] task={task} env=urbanflow model={MODEL_NAME}")

    try:
        while True:
            action = get_action(obs)
            obs, reward, done, info = env.step(action)

            step_count += 1
            rewards.append(f"{reward:.2f}")

            error = info.get("error", "null")

            print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")

            if done:
                success = reward > 0.5
                break

    except Exception as e:
        print(f"[STEP] step={step_count} action=null reward=0.00 done=true error={str(e)}")

    finally:
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}")


if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run_task(t)