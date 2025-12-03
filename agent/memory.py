from typing import Any

class ExperienceReplay:
    # (s, a, r, s', terminal)
    def __init__(self, mem_size: int = 10000) -> None:
        self.mem_size: int = mem_size
        self.memory: list[tuple[Any, Any, Any, Any, Any]] = []
    

    def append(self, transition: tuple[Any, Any, Any, Any, Any]) -> None:
        self.memory.append(transition)

        while len(self.memory) > self.mem_size:
            self.memory.pop(0)