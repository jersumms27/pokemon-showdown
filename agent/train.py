from .model import DQN
from .memory import ExperienceReplay
from .state import Transition
from .player import BotBoi

from poke_env.player import Player

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
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 train_freq: int = 4,
                 target_update_freq: int = 1000
    ) -> None:
        self.agent: BotBoi = agent
        self.opponent: Player = opponent
        self.memory: ExperienceReplay = memory
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: torch.device = device

        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.train_freq: int = train_freq
        self.target_update_freq: int = target_update_freq

        self.train_steps: int = 0
        self.target_model: DQN = deepcopy(agent.model).to(device)
        self.target_model.eval()
    

    async def train_model(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            await self.run_episode()
            self.update_model()


    async def run_episode(self) -> None:
        await self.agent.battle(self.opponent, self.memory)


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