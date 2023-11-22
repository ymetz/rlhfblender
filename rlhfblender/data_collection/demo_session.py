"""
To generate demos, we initialize a gym environment in a separate process, then communicate with it via
a socket. This function initializes the environment via a gym id, optional seed and returns the port

Implements the methods: create_demo_session, demo_step, close_demo_session


"""

import asyncio
import os
import socket
import sys
from multiprocessing import Process
from typing import List, Union

import cv2
import gymnasium as gym
import numpy as np

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


async def create_new_session(session_id: str, gym_env: str, seed: Union[str, int]):
    """
    Create a new session as a asynchronous process. In the process, initialize a gym environment and
    wait for commands via a pipe.
    :param session_id:
    :param demo_number:
    :param gym_env:
    :param seed:
    :return:
    """
    # Create render directory if it doesn't exist
    os.makedirs(os.path.join("data", "current_demos"), exist_ok=True)
    os.makedirs(os.path.join("data", "generated_demos"), exist_ok=True)

    # Check if demos with this session id already exist, find out the demo number (TODO: this is not very efficient)
    demo_number = 0
    while os.path.exists(os.path.join("data", "generated_demos", session_id + "_" + str(demo_number) + ".npz")):
        demo_number += 1

    # Create a new process
    p = Process(target=run_env_session, args=(session_id, demo_number, gym_env, seed))
    p.start()

    # Wait for the process to be ready
    while not os.path.exists(os.path.join("/tmp", session_id + "-port")):
        await asyncio.sleep(0.1)

    return p.pid, demo_number


def run_env_session(session_id: str, demo_number: int, gym_env: str, seed: Union[str, int]):
    """
    Blocking loop that initializes a gym environment and waits for commands via a socket.
    :param session_id: The unique id of the session
    :param demo_number: The index of the demo
    :param gym_env: The gym environment id
    :param seed: The seed for the environment
    :return:
    """
    # Create the gym environment
    env = gym.make(gym_env, render_mode="rgb_array")

    obs_buffer = []
    rew_buffer = []
    done_buffer = []
    info_buffer = []
    action_buffer = []

    env_init = False

    # Create the socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of the socket
        s.bind((HOST, PORT))
        s.listen()
        # Timeout after 10 minutes
        s.settimeout(600)
        port = s.getsockname()[1]
        print("Listening on port", port)
        # Write the port to a file so the parent process can read it
        with open(os.path.join("/tmp", session_id + "-port"), "w") as f:
            f.write(str(port))

        # Wait for commands
        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                # convert the data to a dict
                data = eval(data.decode("utf-8"))
                assert isinstance(data, dict)
                if not data:
                    break
                elif data["command"] == "step":
                    if not env_init:
                        obs = env.reset(seed=seed)
                        obs_buffer.append(obs)
                        env_init = True
                        render = env.render()
                        reward = 0
                        done = False
                        if "BabyAI" not in gym_env:
                            info = {}
                        else:
                            # Such that first frame of demo modal also shows the mission
                            info = {"mission": obs["mission"]}
                    else:
                        observation, reward, terminated, truncated, info = env.step(data["action"])
                        done = terminated or truncated
                        if data["action"] == 6:
                            # 6 is the "done" action in BabyAI
                            done = True
                        rew_buffer.append(reward)
                        done_buffer.append(done)
                        info_buffer.append(info)
                        action_buffer.append(data["action"])
                        if done is True:
                            # Save the episode by first converting the buffers to numpy arrays
                            obs_buffer = np.array(obs_buffer)
                            rew_buffer = np.array(rew_buffer)
                            done_buffer = np.array(done_buffer)
                            action_buffer = np.array(action_buffer)
                            info_buffer = np.array(info_buffer)
                            # Get filename of existing generated demos
                            with open(
                                os.path.join(
                                    "data",
                                    "generated_demos",
                                    session_id + "_" + str(demo_number) + ".npz",
                                ),
                                "wb",
                            ) as f:
                                np.savez(
                                    f,
                                    obs=obs_buffer,
                                    rewards=rew_buffer,
                                    dones=done_buffer,
                                    actions=action_buffer,
                                    infos=info_buffer,
                                )
                        else:
                            obs_buffer.append(obs)
                        render = env.render()

                    # Save the render as an image
                    first_step_render = render
                    first_step_render = cv2.convertScaleAbs(first_step_render)
                    first_step_render = cv2.cvtColor(first_step_render, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join("data", "current_demos", f"{session_id}.jpg"),
                        first_step_render,
                    )

                    return_data = {"reward": reward, "done": done, "infos": info}
                    conn.sendall(str(return_data).encode("utf-8"))
                elif data["command"] == "close":
                    env.close()
                    # Close the socket
                    conn.sendall(str("closed").encode("utf-8"))
                    break

    # Exit the process
    sys.exit(0)


def demo_perform_step(session_id: str, action: Union[int, List[float]]) -> dict:
    """
    Send a step command via the socket to the environment and return the results
    :param session_id:
    :param action:
    :return:
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # Send the step command
        s.sendall(str({"command": "step", "action": action}).encode("utf-8"))

        # Wait for the results
        data = s.recv(1024).decode("utf-8")

        print(data)

        # convert the data to a dict
        data = eval(data)

        return data


def close_demo_session(session_id: str, pid: int):
    """
    Send a close command via the socket to the environment and return the results. Also terminate the process.
    :param session_id:
    :param pid:
    :return:
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # Send the close command
        s.sendall(str({"command": "close"}).encode("utf-8"))

        # Wait for the status
        data = s.recv(1024).decode("utf-8")
        if data == "closed":
            print("Closed session", session_id)
        else:
            print("Error closing session", session_id)

    # Remove the port file
    os.remove(os.path.join("/tmp", session_id + "-port"))


def check_socket_connection(session_id: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            return True
        except:
            return False
