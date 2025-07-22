#!/usr/bin/env python3
"""
Example: Joint Observation-State Projection Usage

This script demonstrates how to:
1. Train a joint projection that maps observations <-> coordinates <-> states
2. Use the trained model to generate demos from novel coordinate clicks
3. Load states in WebRTC demo sessions for interactive exploration

Usage Examples:

# Train joint projection on multiple checkpoints
python example_joint_projection_usage.py train \
  --experiment-id 123 \
  --checkpoints 100000 200000 300000 \
  --environment metaworld_reach-v2 \
  --output-dir models/metaworld_joint

# Use trained model to predict state from coordinates
python example_joint_projection_usage.py predict \
  --model-dir models/metaworld_joint \
  --coordinate 0.5 -1.2

# Generate WebRTC demo session from coordinate
python example_joint_projection_usage.py demo \
  --model-dir models/metaworld_joint \
  --coordinate 0.8 0.3 \
  --session-id test_session
"""

import argparse
import logging
import numpy as np
import json
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_joint_projection(args):
    """Train joint observation-state projection."""
    print("🚀 Training joint observation-state projection...")
    print(f"   Experiment ID: {args.experiment_id}")
    print(f"   Checkpoints: {args.checkpoints}")
    print(f"   Environment: {args.environment}")
    print(f"   Output directory: {args.output_dir}")
    
    # Import here to avoid loading heavy dependencies if not needed
    from scripts.train_joint_observation_state_projection import JointObservationStateProjection
    import asyncio
    
    async def run_training():
        trainer = JointObservationStateProjection(
            projection_method=args.projection_method,
            state_network_params={
                'hidden_dims': [128, 256, 256, 128],
                'learning_rate': 0.001,
                'num_epochs': args.epochs,
                'batch_size': args.batch_size
            }
        )
        
        results = await trainer.fit_joint_projection(
            experiment_id=args.experiment_id,
            checkpoints=args.checkpoints,
            environment_name=args.environment,
            max_trajectories_per_checkpoint=args.max_trajectories,
            max_steps_per_trajectory=args.max_steps
        )
        
        model_paths = trainer.save_joint_model(args.output_dir)
        return results, model_paths
    
    results, model_paths = asyncio.run(run_training())
    
    print("✅ Training completed!")
    print(f"   Forward projection: {model_paths['forward_path']}")
    print(f"   Inverse state projection: {model_paths['inverse_path']}")
    
    # Print consistency metrics
    consistency = results['consistency_metrics']
    print(f"   Overall reconstruction MSE: {consistency['overall_mse']:.6f}")
    
    return model_paths


def predict_state_from_coordinate(args):
    """Predict environment state from 2D coordinate."""
    print(f"🔮 Predicting state from coordinate {args.coordinate}...")
    
    # Load joint model
    from scripts.train_joint_observation_state_projection import JointObservationStateProjection
    
    trainer = JointObservationStateProjection()
    trainer.load_joint_model(args.model_dir)
    
    # Make prediction via API (coordinates -> state)
    response = requests.post(
        "http://localhost:8000/projection/coordinates_to_states",
        json={
            "coordinates": [args.coordinate],
            "model_path": str(Path(args.model_dir) / "inverse_state_projection.pkl")
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        predicted_state = result["states"][0]
        
        print("✅ State prediction successful!")
        print(f"   Coordinate: {args.coordinate}")
        print(f"   Predicted state components:")
        for key, value in predicted_state.items():
            if isinstance(value, list):
                value_str = f"array{np.array(value).shape}"
            else:
                value_str = str(value)
            print(f"     {key}: {value_str}")
        
        return predicted_state
    else:
        print(f"❌ Prediction failed: {response.text}")
        return None


def start_webrtc_demo_from_coordinate(args):
    """Start WebRTC demo session from coordinate."""
    print(f"🎮 Starting WebRTC demo from coordinate {args.coordinate}...")
    
    # Prepare WebRTC offer with coordinate and state model
    webrtc_request = {
        "sdp": "dummy_sdp",  # This would be real SDP in practice
        "type": "offer",
        "session_id": args.session_id,
        "experiment_id": args.experiment_id or "default",
        "environment_id": args.environment or "metaworld_reach-v2",
        "coordinate": args.coordinate,
        "state_model_path": str(Path(args.model_dir) / "inverse_state_projection.pkl")
    }
    
    print("📡 WebRTC demo configuration:")
    print(f"   Session ID: {args.session_id}")
    print(f"   Coordinate: {args.coordinate}")
    print(f"   State model: {webrtc_request['state_model_path']}")
    
    # In practice, you would send this to the WebRTC endpoint
    print("💡 To actually start the demo, send this request to:")
    print("   POST /demo_generation/gym_offer")
    print(f"   Body: {json.dumps(webrtc_request, indent=2)}")
    
    return webrtc_request


def list_available_models(args):
    """List available trained models."""
    print("📋 Available joint projection models:")
    
    # Check local models directory
    models_base = Path("data/saved_projections/models/states")
    if models_base.exists():
        local_models = list(models_base.glob("*.pkl"))
        print(f"   Local models ({len(local_models)}):")
        for model in local_models:
            print(f"     {model}")
    
    # Check via API
    try:
        response = requests.get("http://localhost:8000/projection/available_state_models")
        if response.status_code == 200:
            api_models = response.json()["available_models"]
            print(f"   API models ({len(api_models)}):")
            for model in api_models:
                print(f"     {model}")
        else:
            print("   API not available")
    except:
        print("   API not reachable")


def main():
    parser = argparse.ArgumentParser(description="Joint Observation-State Projection Example")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train joint projection')
    train_parser.add_argument("--experiment-id", required=True, help="Database experiment ID")
    train_parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                             help="Checkpoint steps to include")
    train_parser.add_argument("--environment", required=True, help="Environment name")
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument("--projection-method", default="UMAP", choices=["UMAP", "PCA", "TSNE"])
    train_parser.add_argument("--max-trajectories", type=int, default=50)
    train_parser.add_argument("--max-steps", type=int, default=200)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict state from coordinate')
    predict_parser.add_argument("--model-dir", required=True, help="Directory with trained models")
    predict_parser.add_argument("--coordinate", nargs=2, type=float, required=True,
                               help="2D coordinate [x y]")
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Start WebRTC demo from coordinate')
    demo_parser.add_argument("--model-dir", required=True, help="Directory with trained models")
    demo_parser.add_argument("--coordinate", nargs=2, type=float, required=True,
                            help="2D coordinate [x y]")
    demo_parser.add_argument("--session-id", required=True, help="WebRTC session ID")
    demo_parser.add_argument("--experiment-id", help="Database experiment ID")
    demo_parser.add_argument("--environment", help="Environment name")
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_joint_projection(args)
    elif args.command == 'predict':
        predict_state_from_coordinate(args)
    elif args.command == 'demo':
        start_webrtc_demo_from_coordinate(args)
    elif args.command == 'list':
        list_available_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()