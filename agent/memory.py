import random
from collections import deque
from psycopg2.extensions import connection

from .state import Transition
from db.connection import get_best_transitions

class ExperienceReplay:
    # (s, a, r, s', terminal)
    def __init__(self, mem_size: int = 10000, conn: connection | None = None) -> None:
        self.mem_size: int = mem_size
        self.memory: deque[Transition] = deque([])
    

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

        while len(self.memory) > self.mem_size:
            self.memory.popleft()
        
    
    def get_random_sample(self, sample_size: int) -> list[Transition]:
        return random.sample(self.memory, min(sample_size, len(self.memory)))
    

    def __len__(self) -> int:
        return len(self.memory)