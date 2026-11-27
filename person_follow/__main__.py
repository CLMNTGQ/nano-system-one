import argparse
import json
from pathlib import Path
from .model import Network
from .policy import observations, replay

parser = argparse.ArgumentParser(description="Replay one person-follow-v1 observation episode")
parser.add_argument("episode")
parser.add_argument("--model",type=Path,help="Optional MLP or Transformer checkpoint")
args = parser.parse_args()
model = Network.load(args.model) if args.model else None
for decision,_ in replay(observations(args.episode),model):
    print(json.dumps(decision))
