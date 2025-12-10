from typing import Any
import json

from .model import DQN
from .memory import ExperienceReplay
from .state import Transition
from .player import BotBoi
import db.connection as db

from poke_env.player import Player
from psycopg2.extensions import connection
from datetime import datetime

import torch
from torch import Tensor
import torch.nn.functional as F

from copy import deepcopy

class DQNTrainer:
    def __init__(self,
                 agent: BotBoi,
                 opponent: Player,
                 memory: ExperienceReplay,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 model_checkpoint: str,
                 conn: connection | None,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.99999,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 train_freq: int = 4,
                 target_update_freq: int = 200
    ) -> None:
        self.conn: connection | None = conn
        self.agent: BotBoi = agent
        self.opponent: Player = opponent
        self.memory: ExperienceReplay = memory
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: torch.device = device

        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay

        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.train_freq: int = train_freq
        self.target_update_freq: int = target_update_freq
        self.episode_buffer: list[Transition] = []
        self.checkpoint: str = model_checkpoint

        self.train_steps: int = 0
        self.target_model: DQN = deepcopy(agent.model).to(device)
        self.target_model.eval()

        self.version_id: int = 5 # self.dump_model_to_db(model_checkpoint, lr, notes="Initial model (random initial weights)")
    

    async def train_model(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            self.episode_buffer.clear()

            episode_start: datetime = datetime.now()
            await self.run_episode()
            episode_end: datetime = datetime.now()

            for _ in range(self.train_freq):
                self.update_model()
                self.train_steps += 1

                if self.train_steps % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.agent.model.state_dict())
                    torch.save(self.agent.model.state_dict(), self.checkpoint)

            self.dump_battle_to_db(episode_start, episode_end)
        
        torch.save(self.agent.model.state_dict(), self.checkpoint)


    async def run_episode(self) -> None:
        await self.agent.battle(self.opponent, self.memory, self.episode_buffer, self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)


    def update_model(self) -> None:
        batch: list[Transition] = self.memory.get_random_sample(self.batch_size)

        state_vecs = [torch.from_numpy(t.state.state_vector) for t in batch]
        next_state_vecs = [torch.from_numpy(t.next_state.state_vector) for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        terminals = [t.terminal for t in batch]

        state: Tensor = torch.stack(state_vecs).to(self.device)
        next_state: Tensor = torch.stack(next_state_vecs).to(self.device)
        action: Tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        reward: Tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminal: Tensor = torch.tensor(terminals, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            y: Tensor = reward + self.gamma * torch.max(self.target_model(next_state), dim=1).values * (1.0 - terminal.float())
        y_hat: Tensor = self.agent.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        loss: Tensor = F.mse_loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def dump_battle_to_db(self, ep_start_time: datetime, ep_end_time: datetime) -> None:
        if self.conn is None:
            return
        
        episode_id: int = db.insert_episode(self.conn, self.version_id, ep_start_time, ep_end_time)

        for step_index, t in enumerate(self.episode_buffer):
            state_id: int = db.insert_battle(self.conn, t.state.state_dict)
            new_state_id: int = db.insert_battle(self.conn, t.next_state.state_dict)

            db.insert_transition(self.conn, step_index, episode_id, state_id, new_state_id, t.action, t.reward, t.terminal)
        
        self.conn.commit()


    def dump_model_to_db(self, checkpoint_path: str, lr: float, notes: str = "") -> int:
        if self.conn is None:
            return -1
        
        metrics: dict[str, Any] = {"gamma": self.gamma, "lr": lr, "epsilon_decay": self.epsilon_decay, "notes": notes}
        metrics_json: str = json.dumps(metrics)

        version_id = db.insert_model(self.conn, datetime.now().date(), checkpoint_path, metrics_json)
        self.conn.commit()

        return version_id