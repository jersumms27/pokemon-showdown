# Project Purpose and Goals
**A stateful reinforcement learning system that trains a Pokémon Showdown battle agent while storing all battle data, model versions, and evaluation results in a Postgres database.**

The purpose of this project is to create a reinforcement learning (RL) agent that learns how to battle in Pokemon Showdown optimally. Training an agent using reinforcement learning requires a memory of previous states that the agent has been in, the action it took from that state, and the reward it received for transitioning to the next state. For easy accessibility and long-term storage, these data will be stored in a tabular database. Information about the model's performance and hyperparameters will also be stored to track the agent's learning.

In Pokemon Showdown, at any given moment during a battle, the battle's state can be represented by a battle log. The log contains information about what moves each player has taken as well as the results and effects each move had. This will be the main source of information when creating game states and rewards. Eventually, these logs will also be embedded and stored as vectors. This way, as the agent is battling, it can compare the current battle log against other logs through vector search to see if it has ever been in a similar battle before.

Instead of keeping battles and training data in memory, the database persists episodes, transitions, and model versions so training can resume, analytics can run independently, and the system behaves like a real stateful ML service rather than a single script.

# ERD
(See Github repo - `erd.png`.)

# System Design
(See Github repo - `system_design.png`.)

# Initial Goals
* **Week 1 (3 Nov - 9 Nov)**
    * Finalize ERD
    * Set up Postgres schema
    * Research about Pokemon Showdown API
* **Week 2 (10 Nov - 16 Nov)**
    * Code up simple RL agent
    * Create reward function
    * Set up functionality to store transitions in DB
* **Week 3 (17 Nov - 23 Nov)**
    * Finalize database setup
    * Start initial model training
    * Store model versions and performance results in DB
* **Week 4 (24 Nov - 30 Nov)**
    * Implement vector search
    * Continue model training and optimization
* **Week 5 (1 Dec - 7 Dec)**
    * Continue model training and optimization
    * Begin creation of presentation
* **Week 6\* (8 Dec - 10 Dec)**
    * Finalize presentation