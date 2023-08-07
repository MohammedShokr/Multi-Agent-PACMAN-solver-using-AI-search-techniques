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


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from math import inf

class ReflexAgent(Agent):
	"""
	Question 1: Reflex Agent
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.
	"""


	def getAction(self, gameState):
		"""
		getAction chooses among the best options according to the evaluation function.

		GetAction takes a GameState and returns some Directions.X for some X in the
		set {NORTH, SOUTH, WEST, EAST, STOP}
		"""

		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()
		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.
		"""

		#Extract useful information about the current state and the next state to be used to evaluate the state
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		ghostsPosition = [ghostState.getPosition() for ghostState in newGhostStates]
		#Calculating the distance between PACMAN and all the ghosts
		min_ghostsDistance=min([manhattanDistance(newPos,ghost_Pos) for ghost_Pos in ghostsPosition])
		#Calculating the distance between PACMAN and all food in the grid
		foodDistances=[manhattanDistance(newPos,food_pos) for food_pos in newFood.asList()]
		#Evaluating the state
		if min_ghostsDistance==1:
			min_ghostsDistance=-999

		if foodDistances: 
			min_foodDistance=min(foodDistances)
		else: 
			min_foodDistance=0
		return 0.25*min_ghostsDistance+10*successorGameState.getScore()-min_foodDistance

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
	Question 2: Minimax Agent
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.
		"""
		
		#Define PACMAN index
		PACMAN = 0
		def max_value(state, depth):
			#First check if it is winning state, losing state or terminal state
			if state.isWin() or state.isLose():
				return state.getScore()
			if depth == self.depth or len(state.getLegalActions(PACMAN))==0:
				return(self.evaluationFunction(state))
			best_maxi = -inf
			maxi = best_maxi
			best_action = Directions.STOP
			#Loop on all posible actions and choose the best action based on Min agent decision
			for action in state.getLegalActions(PACMAN):
				#Call the first ghost in the same depth 
				maxi = min_value(state.generateSuccessor(PACMAN, action), depth, 1)
				if maxi > best_maxi:
					best_maxi = maxi
					best_action = action
			if depth == 0:
				return best_action
			else:
				return best_maxi

		def min_value(state, depth, ghost_idx):
			#First check if it is winning state, losing state or terminal state
			if state.isLose() or state.isWin():
				return state.getScore()
			if len(state.getLegalActions(ghost_idx))==0:
				return(self.evaluationFunction(state))            
			next_ghost_idx = ghost_idx + 1
			if ghost_idx == state.getNumAgents() - 1:
				next_ghost_idx = PACMAN
			best_mini = +inf
			mini = best_mini
			#Loop on all posible actions and choose the best action based on Max agent decision
			for action in state.getLegalActions(ghost_idx):
				if next_ghost_idx == PACMAN:
					mini = max_value(state.generateSuccessor(ghost_idx, action), depth + 1)
				else:
					#Each agent calls the function for all next agents till the next agent that calls PACMAN
					mini = min_value(state.generateSuccessor(ghost_idx, action), depth, next_ghost_idx)
				if mini < best_mini:
					best_mini = mini
			return best_mini
		return max_value(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Question 3: Alpha-Beta pruning
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		PACMAN = 0
		def max_value(state, depth, alpha, beta):
			if state.isWin() or state.isLose():
				return state.getScore()
			if depth == self.depth or len(state.getLegalActions(PACMAN))==0:
				return(self.evaluationFunction(state))
			best_maxi = -inf
			maxi = best_maxi
			best_action = Directions.STOP
			for action in state.getLegalActions(PACMAN):
				maxi = min_value(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
				if maxi > best_maxi:
					best_maxi = maxi
					best_action = action
					#Update Alpha
					if best_maxi > beta: return best_maxi
					alpha = max(alpha,best_maxi)
			if depth == 0:
				return best_action
			else:
				return best_maxi

		def min_value(state, depth, ghost_idx, alpha, beta):
			if state.isLose() or state.isWin():
				return state.getScore()
			if len(state.getLegalActions(ghost_idx))==0:
				return(self.evaluationFunction(state))            
			next_ghost_idx = ghost_idx + 1
			if ghost_idx == state.getNumAgents() - 1:
				next_ghost_idx = PACMAN
			best_mini = +inf
			mini = best_mini
			for action in state.getLegalActions(ghost_idx):
				if next_ghost_idx == PACMAN:
					mini = max_value(state.generateSuccessor(ghost_idx, action), depth + 1, alpha, beta)
				else:
					mini = min_value(state.generateSuccessor(ghost_idx, action), depth, next_ghost_idx, alpha, beta)
				if mini < best_mini:
					best_mini = mini
					#Update beta
					if best_mini < alpha: return best_mini
					beta = min(beta,best_mini)
			return best_mini
		return max_value(gameState, 0, -inf, +inf)

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Question 4: Expectimax
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts are modeled as choosing uniformly at random from their legal moves.
		"""

		PACMAN = 0
		#Same max agent as Minimax
		def max_value(state, depth):
			if state.isWin() or state.isLose():
				return state.getScore()
			if depth == self.depth or len(state.getLegalActions(PACMAN))==0:
				return(self.evaluationFunction(state))
			best_maxi = -inf
			maxi = best_maxi
			best_action = Directions.STOP
			for action in state.getLegalActions(PACMAN):
				maxi = min_value(state.generateSuccessor(PACMAN, action), depth, 1)
				if maxi > best_maxi:
					best_maxi = maxi
					best_action = action
			if depth == 0:
				return best_action
			else:
				return best_maxi

		def min_value(state, depth, ghost_idx):
			if state.isLose() or state.isWin():
				return state.getScore()
			if len(state.getLegalActions(ghost_idx))==0:
				return(self.evaluationFunction(state))            
			next_ghost_idx = ghost_idx + 1
			if ghost_idx == state.getNumAgents() - 1:
				next_ghost_idx = PACMAN
			mini = 0
			#Calculating the probability of uniformly distributed actions
			p=1.0/len(state.getLegalActions(ghost_idx))
			for action in state.getLegalActions(ghost_idx):
				#Calculating the expected mini value not the optimal one
				if next_ghost_idx == PACMAN:
					mini += max_value(state.generateSuccessor(ghost_idx, action), depth + 1)
				else:
					mini += min_value(state.generateSuccessor(ghost_idx, action), depth, next_ghost_idx)
			return p*mini
		return max_value(gameState, 0)

def betterEvaluationFunction(currentGameState):
	"""
	Question 5: Designing Evaluation function

	DESCRIPTION: <This function is meant to evaluate the terminal states. It evaluates the state 
	based on the distance between PACMAN and the nearest ghost, the distance between PACMAN and the nearest food
	, and the distance between PACMAN and the nearest capsule. It returns a weighted sum of all the mentioned features.>
	"""
	score=0
	#Extracting the needed information about the next and current states
	newPos = currentGameState.getPacmanPosition()
	newFood = currentGameState.getFood()
	newGhostStates = currentGameState.getGhostStates()
	ghostsPosition = [ghostState.getPosition() for ghostState in newGhostStates]
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	newCapsules=currentGameState.getCapsules()
	#Measuring the needed distances
	min_ghostsDistance=min([manhattanDistance(newPos,ghost_Pos) for ghost_Pos in ghostsPosition])
	foodDistances=[manhattanDistance(newPos,food_pos) for food_pos in newFood.asList()]
	capsuleDistances=[manhattanDistance(newPos,food_pos) for food_pos in newCapsules]

	#Edit the score based on the nearest ghost distance and the scared time 
	if min_ghostsDistance<=1:
		if newScaredTimes:
			score += 999
		else:
			score -= 999
	else:
		if newScaredTimes:
			if min(newScaredTimes)>=10:
				score += 999
		else:
			score += min_ghostsDistance

	#increase the score with the reciprocal of the food distance
	if foodDistances: 
		min_foodDistance=min(foodDistances)
		if min_foodDistance==0: min_foodDistance=1
		score += 1/min_foodDistance

	#Increase the score with the score PACMAN in this state 
	score += currentGameState.getScore()

	#Edit the score based on the nearest capsule
	if capsuleDistances:
		min_capsuleDistance=min(capsuleDistances)
		if min_capsuleDistance<=1:
			score = 999
		else: 
			score += 5/min_capsuleDistance 

	return score

# Abbreviation
better = betterEvaluationFunction
