# AI-Powered 3D Traveling Salesman Problem Solver

This project implements a solution for a 3D Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA). The problem is framed in the context of a USC student optimizing travel routes across a series of 3D coordinates representing campus locations. The objective is to identify the shortest possible round-trip path that visits each location exactly once and returns to the starting point.

## Features

- **Initial Population Creation**: Generates an initial set of candidate solutions (paths) by randomly or heuristically arranging city coordinates.
- **Parent Selection**: Uses a roulette-wheel selection mechanism to choose parent paths for crossover, ensuring higher selection probability for fitter candidates.
- **Crossover Operation**: Combines two parent paths using a two-point crossover method to produce a valid child path that adheres to TSP constraints.
- **Mutation and Fitness Evaluation**: Introduces random variations and evaluates the fitness of paths based on the total distance traveled, favoring shorter paths.
- **Output Specification**: Outputs the shortest computed distance and the corresponding sequence of 3D coordinates visited in a valid loop format.

## Input and Output Format

- **Input**: The program reads input from a file named `input.txt`, containing:
  1. The number of locations (a strictly positive integer).
  2. A list of 3D coordinates (x, y, z) for each location.

- **Output**: The program generates an `output.txt` file containing:
  1. The total computed distance of the path.
  2. The sequence of 3D coordinates visited in order, ending with the starting location.


## How It Works

1. **Input Parsing**: Reads and validates the input data.
2. **Genetic Algorithm Execution**:
 - Initializes a population of candidate paths.
 - Iteratively improves the population through selection, crossover, mutation, and fitness evaluation.
3. **Path Optimization**: Identifies the path with the shortest distance from the final population.
4. **Output Generation**: Writes the optimal path and its total distance to `output.txt`.

## Requirements

- Python 3.7.5 or compatible
- Standard Python libraries (no additional dependencies)

## Running the Program

1. Place the input file (`input.txt`) in the program's directory.
2. Run the program using the command: python3 hw1.py

 


