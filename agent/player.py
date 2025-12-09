from typing import Awaitable
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import Player
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from .model import DQN
from .state import State

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from numpy import ndarray

class BotBoi(Player):
    def __init__(self, dqn_model: DQN, device: torch.device, **kwargs):
        super().__init__(**kwargs)
    
        self.model: DQN = dqn_model
        self.device: torch.device = device
        self.epsilon: float = 0.1

        self.model.to(device)

    def _get_action_mask(self, battle: AbstractBattle) -> ndarray:
        mask: list[bool] = []
        
        # regular moves
        num_moves: int = len(battle.available_moves)
        move_mask: list[bool] = [True] * num_moves + [False] * (4 - num_moves)
        mask.extend(move_mask)

        # switches
        num_switches: int = len(battle.available_switches)
        switch_mask: list[bool] = [True] * num_switches + [False] * (5 - num_switches)
        mask.extend(switch_mask)

        # mega-evolution
        mask.extend([battle.can_mega_evolve and m for m in move_mask])

        # z-move
        mask.append(battle.can_z_move)

        # dynamax
        mask.extend([battle.can_dynamax and m for m in move_mask])

        # terastallize
        mask.extend([battle.can_tera and m for m in move_mask])

        return np.array(mask, dtype=bool)
    

    def _idx_to_action(self, battle: AbstractBattle, idx: int) -> SingleBattleOrder:
        num_moves: int = len(battle.available_moves)
        num_switches: int = len(battle.available_switches)

        actions: list[tuple[Move | Pokemon | str, bool, bool, bool, bool]] = [
            (battle.available_moves[0] if num_moves >= 1 else "", False, False, False, False),
            (battle.available_moves[1] if num_moves >= 2 else "", False, False, False, False),
            (battle.available_moves[2] if num_moves >= 3 else "", False, False, False, False),
            (battle.available_moves[3] if num_moves >= 4 else "", False, False, False, False),
            (battle.available_switches[0] if num_switches >= 1 else "", False, False, False, False),
            (battle.available_switches[1] if num_switches >= 2 else "", False, False, False, False),
            (battle.available_switches[2] if num_switches >= 3 else "", False, False, False, False),
            (battle.available_switches[3] if num_switches >= 4 else "", False, False, False, False),
            (battle.available_switches[4] if num_switches >= 5 else "", False, False, False, False),
            (battle.available_moves[0] if num_moves >= 1 else "", True, False, False, False),
            (battle.available_moves[1] if num_moves >= 2 else "", True, False, False, False),
            (battle.available_moves[2] if num_moves >= 3 else "", True, False, False, False),
            (battle.available_moves[3] if num_moves >= 4 else "", True, False, False, False),
            (battle.available_moves[0] if num_moves >= 1 else "", False, True, False, False),
            (battle.available_moves[0] if num_moves >= 1 else "", False, False, True, False),
            (battle.available_moves[1] if num_moves >= 2 else "", False, False, True, False),
            (battle.available_moves[2] if num_moves >= 3 else "", False, False, True, False),
            (battle.available_moves[3] if num_moves >= 4 else "", False, False, True, False),
            (battle.available_moves[0] if num_moves >= 1 else "", False, False, False, True),
            (battle.available_moves[1] if num_moves >= 2 else "", False, False, False, True),
            (battle.available_moves[2] if num_moves >= 3 else "", False, False, False, True),
            (battle.available_moves[3] if num_moves >= 4 else "", False, False, False, True),
        ]

        return SingleBattleOrder(*actions[idx])
    

    def choose_move(self, battle: AbstractBattle) -> BattleOrder | Awaitable[BattleOrder]:
        state: State = State(battle)
        state_vec: Tensor = torch.tensor(state.state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            q_values: Tensor = self.model(state_vec)[0].cpu().numpy()
        
        mask: ndarray = self._get_action_mask(battle)
        q_values[~mask] = -np.inf

        if np.random.rand() < self.epsilon:
            legal_idx: ndarray = np.where(mask)[0]
            idx = int(np.random.choice(legal_idx))
        else:
            idx = int(q_values.argmax())
        
        action: SingleBattleOrder = self._idx_to_action(battle, idx)
        return action