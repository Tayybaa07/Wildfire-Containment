"""Wildfire containment MDP (CS272 project) - Stage 1: environment and baselines."""
from .config import Config, make_config, make_vegetation, U, B, X, F, GRASS, SHRUB, FOREST
from .env import WildfireEnv
from .agents import (Agent, RandomAgent, NearestFrontAgent, GreedyAgent,
                     NoOpAgent, AGENTS)
from .simulate import run_episode, evaluate, mean_ci
