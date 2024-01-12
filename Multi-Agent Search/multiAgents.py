# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


#Betül Dinçer-64750

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
 

        "*** YOUR CODE HERE ***"
        curr_score = successorGameState.getScore()
        dist_to_ghost = manhattanDistance(newPos,newGhostStates[0].getPosition())
        if dist_to_ghost >0:
            curr_score -= 8/dist_to_ghost

        foods = newFood.asList()
        distanceList = []
        if len(foods)>0:

            for food in foods:
                distanceList=distanceList+[manhattanDistance(newPos, food)]
            curr_score += 5/min(distanceList)
        sum_ScaredTime = sum(newScaredTimes) 
        if sum_ScaredTime!=0:
            curr_score-=4/sum_ScaredTime

        return curr_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        actions, value = self.mini_max(0, 0, gameState)  # for pacman the actşons and the value
        return actions   
   
        util.raiseNotDefined()
        
        
    def mini_max(self, depth_tree, agent_index, gameState):

        num_agents=gameState.getNumAgents(); #if agent index exceeds the total number of agents in the game indicates that every player played their turn so the next player is again pacman. And increase the depth of tree 
        if agent_index >= num_agents: 
            agent_index = 0
            depth_tree = depth_tree + 1
     
        current_value=-999 #initialize the next score and action
        next_action = "West"
        iteration=0 #to change best score in first round

        if agent_index == 0:  # pacman is playing, the aim is to reach best score and best action
            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of pacman

                transition_func = gameState.generateSuccessor(agent_index, action) # transition function, generates next game state if the agent takes action in legal actions

                dummy, value = self.mini_max(depth_tree, agent_index + 1, transition_func)#calculates ghost's minimax 
                # we dont need action of ghost so I leave it as dummy
                if  value > current_value: #take the greater score
                    current_value = value
                    next_action = action
        elif agent_index!=0:  # ghost is playing
            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState) 
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of ghost agent
                next_game_state = gameState.generateSuccessor(agent_index, action)
                dummy, value = self.mini_max(depth_tree, agent_index + 1, next_game_state)
                if iteration==0 or value < current_value:
                    current_value = value
                    next_action = action
                    iteration=iteration+1

        # if we cannot find any successor states return the value of evaluationFunction of state
        if current_value == -999:
           return None, self.evaluationFunction(gameState)
        return next_action, current_value  # Return action and the value
    
    
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        actions, value = self.alpha_beta_pruning(0, 0, gameState, -9999, 9999)  # for pacman
        return actions 

    def alpha_beta_pruning(self, depth_tree, agent_index, gameState, alpha, beta):

        num_agents=gameState.getNumAgents(); #if agent index exceeds the total number of agents in the game indicates that every player played their turn so the next player is again pacman. And increase the depth of tree 
        if agent_index >= num_agents: 
            agent_index = 0
            depth_tree = depth_tree + 1
     
        current_value=-999 #initialize the next score and action
        next_action = "West"
        iteration=0 #to change best score in first round

        if agent_index == 0:  # pacman is playing, the aim is to reach best score and best action
            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of pacman

                transition_func = gameState.generateSuccessor(agent_index, action) # transition function, generates next game state if the agent takes action in legal actions

                dummy, value = self.alpha_beta_pruning(depth_tree, agent_index + 1, transition_func,alpha,beta)#calculates ghost's minimax 
                # we dont need action of ghost so I leave it as dummy
                if  value > current_value: #take the greater score
                    current_value = value
                    next_action = action
                    
                
                if alpha<value:
                   alpha=value 
                # Prune the tree if alpha is greater than beta
                if alpha > beta:
                    break
                    
                    
        elif agent_index!=0:  # ghost is playing
            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState) 
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of ghost agent
                next_game_state = gameState.generateSuccessor(agent_index, action)
                dummy, value = self.alpha_beta_pruning(depth_tree, agent_index + 1, next_game_state,alpha,beta)
                if iteration==0 or value < current_value:
                    current_value = value
                    next_action = action
                    iteration=iteration+1
                    
                
                if beta>value:
                    beta=value
                # Prune the tree if beta is less than alpha
                if beta < alpha:
                    break
        # we cannot find any successor states return the value of evaluationFunction of state
        if current_value == -999:
           return None, self.evaluationFunction(gameState)
        return next_action, current_value  # Return actions and value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions, value = self.exp(0, 0, gameState)  # for pacman 
        return actions   
   
        util.raiseNotDefined()
        
        
    def exp(self, depth_tree, agent_index, gameState):

        num_agents=gameState.getNumAgents(); #if agent index exceeds the total number of agents in the game indicates that every player played their turn so the next player is again pacman. And increase the depth of tree 
        if agent_index >= num_agents: 
            agent_index = 0
            depth_tree = depth_tree + 1
     
        current_value=-999 #initialize the next score and action
        next_action = "West"
        iteration=0 #to change best score in first round

        if agent_index == 0:  # pacman is playing, the aim is to reach best score and best action
            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of pacman

                transition_func = gameState.generateSuccessor(agent_index, action) # transition function, generates next game state if the agent takes action in legal actions

                dummy, value = self.exp(depth_tree, agent_index + 1, transition_func)#calculates ghost's minimax 
                # we dont need action of ghost so I leave it as dummy
                if  value > current_value: #take the greater score
                    current_value = value
                    next_action = action
        elif agent_index!=0:  # ghost is playing

            if depth_tree == self.depth: #if we reached the max depth, return the score of evaluation function
                return 0, self.evaluationFunction(gameState) 
            actions=gameState.getLegalActions(agent_index)
            for action in actions :  # For each legal action of ghost agent
                transition_func = gameState.generateSuccessor(agent_index, action)
                dummy, value = self.exp(depth_tree, agent_index + 1, transition_func)
                if iteration==0 :
                    current_value=0.0
                    iteration=iteration+1
                    
                #the minimizer(ghost) acting probabilistically
                current_value = current_value+value/len(gameState.getLegalActions(agent_index))
                next_action = action
                    

        # if we cannot find any successor states return the value of evaluationFunction of state
        if current_value == -999:
           return None, self.evaluationFunction(gameState)
        return next_action, current_value  # Return actions and value
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: < this parts calculate evaluation function using ghosts distannce, scared times of ghosts, winning or losing state, and distance to food and capsule
    
    """
    "*** YOUR CODE HERE ***"
    
    
    ghost_states = currentGameState.getGhostStates() #all the ghost states
    position_pac = currentGameState.getPacmanPosition()
    foods = (currentGameState.getFood()).asList() #get all the food as list.
    capsules = currentGameState.getCapsules() #get all the capsules.
    GhostScaredTimes = [ghostState.scaredTimer for ghostState in ghost_states]
    

    evaluation_score = 0 #initially 0.

    #Calculating evaluation score by using distance to ghost and scares time of ghost
    if currentGameState.getNumAgents() > 1:
        ghost_dis = min( [manhattanDistance(position_pac, ghost.getPosition()) for ghost in ghost_states])
        if (ghost_dis <= 1):
            return -10000
        evaluation_score -= 1.0/ghost_dis
        if(GhostScaredTimes[0]==0):
            evaluation_score -=2/ min([manhattanDistance(position_pac, ghost.getPosition()) for ghost in ghost_states])

 
    #if losing state return very negative number, if winning state return very positive number      
    if currentGameState.isLose():
        return -100000 + evaluation_score
    elif currentGameState.isWin():
        return 100000 + evaluation_score
    
    #add food list and capsule list and calculate evaluation function through distance to them
    all_bullet = foods + capsules   
    bullet_score = -len(all_bullet)
    bullet_score += 1/min([manhattanDistance( position_pac, food) for food in all_bullet])    
    evaluation_score=evaluation_score+bullet_score

    return evaluation_score

# Abbreviation
better = betterEvaluationFunction