import metaworld
import gymnasium as gym
import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from metaworld.policies import ENV_POLICY_MAP

def create_multi_camera_videos(env_name, output_dir="camera_videos", duration_seconds=2, fps=30):
    """
    Create 2-second videos from each camera angle in Meta-World
    
    Args:
        env_name: Name of the Meta-World environment
        output_dir: Directory to save videos for this environment
        duration_seconds: Duration of each video in seconds
        fps: Frames per second for the videos
    """
    
    # Create output directory for this environment
    env_output_dir = Path(output_dir) / env_name
    env_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Available camera names in Meta-World (including default)
    camera_names = ['default', 'corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV']
    
    # Calculate total frames needed
    total_frames = duration_seconds * fps
    
    # Get policy for this environment to use realistic actions
    policy_cls = ENV_POLICY_MAP.get(env_name)
    
    for camera_name in camera_names:
        try:
            # Create environment with specific camera or default
            if camera_name == 'default':
                # Create environment without specifying camera (uses default)
                env = gym.make('Meta-World/MT1', 
                              env_name=env_name, 
                              render_mode='rgb_array')
            else:
                # Create environment with specific camera
                env = gym.make('Meta-World/MT1', 
                              env_name=env_name, 
                              render_mode='rgb_array', 
                              camera_name=camera_name)
            
            # Initialize policy if available
            policy = policy_cls() if policy_cls else None
            
            # Reset environment
            obs, info = env.reset()
            
            # Get the first frame to determine video dimensions
            frame = env.render()
            height, width = frame.shape[:2]
            
            # Setup video writer
            output_path = env_output_dir / f"{camera_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Render frames
            for frame_idx in range(total_frames):
                # Use policy action if available, otherwise random action
                if policy:
                    action = policy.get_action(obs)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminate, truncate, info = env.step(action)
                
                # Get frame and convert BGR to RGB for OpenCV
                frame = env.render()
                
                # Fix rotation for corner cameras (Mujoco rendering bug)
                if camera_name.startswith('corner'):
                    # Rotate 180 degrees (2 times 90 degrees clockwise)
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to video
                video_writer.write(frame_bgr)
                
                # Reset environment if episode ends
                if terminate or truncate:
                    obs, info = env.reset()
            
            # Release video writer and close environment
            video_writer.release()
            env.close()
            
        except Exception as e:
            print(f"  ✗ Error rendering {camera_name}: {str(e)}")
            continue
    
    return len(camera_names)

def render_all_environments(output_base_dir="metaworld_videos", duration_seconds=2, fps=30):
    """
    Render videos for all Meta-World environments from all camera angles
    
    Args:
        output_base_dir: Base directory to save all videos
        duration_seconds: Duration of each video in seconds
        fps: Frames per second for the videos
    """
    
    # Create base output directory
    Path(output_base_dir).mkdir(exist_ok=True)
    
    # Get all available environments
    available_envs = list(ENV_POLICY_MAP.keys())
    
    print(f"Starting Multi-Environment Multi-Camera Video Rendering...")
    print(f"Environments: {len(available_envs)}")
    print(f"Cameras per environment: 7 (default, corner, corner2, corner3, topview, behindGripper, gripperPOV)")
    print(f"Duration: {duration_seconds} seconds per video")
    print(f"Total videos to create: {len(available_envs) * 7}")
    print(f"Output directory: {output_base_dir}/")
    print("-" * 70)
    
    total_success = 0
    total_videos = 0
    
    # Process each environment
    for env_name in tqdm(available_envs, desc="Processing environments"):
        print(f"\nRendering videos for: {env_name}")
        
        try:
            cameras_rendered = create_multi_camera_videos(
                env_name=env_name,
                output_dir=output_base_dir,
                duration_seconds=duration_seconds,
                fps=fps
            )
            
            print(f"  ✓ Completed {env_name} ({cameras_rendered} cameras)")
            total_success += cameras_rendered
            total_videos += 7  # Expected number of cameras
            
        except Exception as e:
            print(f"  ✗ Failed to process {env_name}: {str(e)}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Rendering Complete!")
    print(f"Successfully created: {total_success}/{total_videos} videos")
    print(f"Environments processed: {len(available_envs)}")
    print(f"Videos saved to: {output_base_dir}/")
    print(f"\nDirectory structure:")
    print(f"  {output_base_dir}/")
    print(f"  ├── env_name_1/")
    print(f"  │   ├── default.mp4")
    print(f"  │   ├── corner.mp4")
    print(f"  │   ├── corner2.mp4")
    print(f"  │   ├── corner3.mp4")
    print(f"  │   ├── topview.mp4")
    print(f"  │   ├── behindGripper.mp4")
    print(f"  │   └── gripperPOV.mp4")
    print(f"  ├── env_name_2/")
    print(f"  │   └── ...")
    print(f"  └── ...")

def render_specific_environments(env_names, output_base_dir="metaworld_videos", duration_seconds=2, fps=30):
    """
    Render videos for specific Meta-World environments
    
    Args:
        env_names: List of environment names to render
        output_base_dir: Base directory to save all videos
        duration_seconds: Duration of each video in seconds
        fps: Frames per second for the videos
    """
    
    # Create base output directory
    Path(output_base_dir).mkdir(exist_ok=True)
    
    # Filter to only available environments
    available_envs = [env for env in env_names if env in ENV_POLICY_MAP]
    unavailable_envs = [env for env in env_names if env not in ENV_POLICY_MAP]
    
    if unavailable_envs:
        print(f"Warning: The following environments are not available: {unavailable_envs}")
    
    print(f"Rendering videos for {len(available_envs)} specific environments...")
    print(f"Environments: {available_envs}")
    print("-" * 70)
    
    for env_name in tqdm(available_envs, desc="Processing environments"):
        print(f"\nRendering videos for: {env_name}")
        
        try:
            cameras_rendered = create_multi_camera_videos(
                env_name=env_name,
                output_dir=output_base_dir,
                duration_seconds=duration_seconds,
                fps=fps
            )
            
            print(f"  ✓ Completed {env_name} ({cameras_rendered} cameras)")
            
        except Exception as e:
            print(f"  ✗ Failed to process {env_name}: {str(e)}")
            continue

def main():
    # Example usage options:
    
    # Option 1: Render all environments (this will take a while!)
    # render_all_environments()
    
    # Option 2: Render specific environments
    specific_envs = ["reach-v3", "pick-place-v3", "push-v3", "drawer-open-v3"]
    render_specific_environments(specific_envs)
    
    # Option 3: Render all environments with custom settings
    # render_all_environments(
    #     output_base_dir="custom_videos",
    #     duration_seconds=3,
    #     fps=24
    # )

if __name__ == "__main__":
    main()