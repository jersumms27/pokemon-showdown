# Project Purpose and Goals
The purpose of this project is to create a reinforcement learning (RL) agent that learns how to battle in Pokemon Showdown optimally. Training an agent using reinforcement learning requires a memory of previous states that the agent has been in, the action it took from that state, and the reward it received for transitioning to the next state. For easy accessibility and long-term storage, these data will be stored in a tabular database. Information about the model's performance and hyperparameters will also be stored to track the agent's learning.

Instead of keeping battles and training data in memory, the database persists episodes, transitions, and model versions so training can resume, analytics can run independently, and the system behaves like a real stateful ML service rather than a single script.

# ERD
(See Github repo.)

# System Design
The RL model will be implemented using Python. The database will be implemented using Postgres.

# Initial Goals
* **Week 1 (3 Nov - 9 Nov)**
    * Finalize ERD
    * Set up Postgres database
    * Research about Pokemon Showdown API
* **Week 2 (10 Nov - 16 Nov)**
    * Code up RL agent
    * Create reward function
    * Set up connection between Python and Postgres
* **Week 3 (17 Nov - 23 Nov)**
    * Finalize database setup
    * Begin model training
* **Week 4 (24 Nov - 30 Nov)**
    * Add Redis caching
    * Continue model training and optimization
* **Week 5 (1 Dec - 7 Dec)**
    * Continue model training and optmization
    * Implement vector similarity search
* **Week 6\* (8 Dec - 10 Dec)**
    * Finalize presentation