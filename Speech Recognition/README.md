# Temporal Reasoning with POMDP: Little Prince & Speech Recognition

The project focuses on solving temporal reasoning problems using a **Partially Observable Markov Decision Process (POMDP)** framework. The project includes solutions for two scenarios: the "Little Prince" navigation task and a speech recognition task that maps phonemes to graphemes.

## Features

### 1. Little Prince Scenario
- **Objective**: Determine the most probable sequence of hidden states for an agent navigating a fictional environment.
- **Input**:
  - State weights for initial probabilities.
  - State-action-state transition weights.
  - State-observation weights.
  - A sequence of observations and actions (e.g., "Apple", "Turnaround").
- **Output**: A sequence of predicted states that the agent likely traversed.

### 2. Speech Recognition Scenario
- **Objective**: Map a sequence of phonemes to the most probable sequence of graphemes using a POMDP model.
- **Input**:
  - Grapheme-phoneme observation weights.
  - Grapheme transition weights.
  - A sequence of phonemes.
- **Output**: A sequence of predicted graphemes corresponding to the phonemes.

### Key Implementations
- **Normalization**: Converts raw weight data into probability distributions for state transitions, observations, and initial states.
- **Viterbi Algorithm**: Used to infer the most probable sequence of hidden states given a sequence of observations and actions.
 
