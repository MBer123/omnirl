from collections.abc import Mapping

import numpy as np


def extract_observation(state):
    """Extract an algorithm-facing observation from an RLCard state."""
    if not isinstance(state, Mapping) or "obs" not in state:
        raise ValueError("RLCard state must be a mapping containing 'obs'.")

    observation = np.asarray(state["obs"])
    if observation.ndim != 1:
        raise ValueError(
            "The RLCard backend currently supports vector observations; "
            f"received shape {observation.shape}."
        )
    return observation.copy()


def validate_player(player_id):
    """Validate the active player against the backend contract."""
    if player_id != 0:
        raise ValueError(
            "The RLCard backend supports only player 0, "
            f"but the environment selected player {player_id}."
        )


def validate_action(state, action):
    """Validate and return an integer environment action."""
    if not isinstance(state, Mapping) or "legal_actions" not in state:
        raise ValueError(
            "RLCard state must contain 'legal_actions' for action validation."
        )

    try:
        action = int(action)
    except (TypeError, ValueError) as error:
        message = f"RLCard action must be an integer, received {action!r}."
        raise ValueError(message) from error

    if action not in state["legal_actions"]:
        legal_actions = list(state["legal_actions"])
        raise ValueError(
            f"RLCard action {action} is illegal; legal actions are {legal_actions}."
        )
    return action


def seed_and_reset(env, seed=None):
    """Seed and reset an RLCard environment."""
    if seed is not None:
        env.seed(seed)
    state, player_id = env.reset()
    validate_player(player_id)
    return state


def terminal_payoff(env):
    """Return the terminal payoff of player 0."""
    payoffs = np.asarray(env.get_payoffs())
    if payoffs.shape != (1,):
        raise ValueError(
            "The RLCard backend expected one terminal payoff, "
            f"received shape {payoffs.shape}."
        )
    return float(payoffs[0])
