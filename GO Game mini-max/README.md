# AI Agent for Little-Go (5x5 Board Game)

This project involves creating an AI agent to play **Little-Go**, a simplified version of the Go board game with a 5x5 grid. The goal is to implement advanced AI techniques, such as Minimax, Alpha-Beta Pruning, and Q-Learning, to compete against various pre-programmed AI agents in staged tournaments. The focus is on applying search, game-playing, and reinforcement learning concepts.
## Features

### Game Rules
- **Players**: Two players, Black and White.
- **Board**: 5x5 grid of intersections (points) where stones are placed.
- **Rules**:
  - **Liberty Rule**: Stones or groups of stones must have at least one liberty (empty point adjacent to them) or will be captured.
  - **KO Rule**: Immediate recapture of a previously captured stone is prohibited.
  - **Komi**: White is awarded 2.5 points as compensation for Black's first-move advantage.

### Winning Criteria
- Highest final score wins.
  - **Black Score**: Count of Black stones on the board.
  - **White Score**: Count of White stones + Komi (2.5).

### AI Techniques
- **Minimax with Alpha-Beta Pruning**: A depth-limited search strategy for making optimal decisions.
