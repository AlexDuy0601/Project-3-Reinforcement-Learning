# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration1()

    def runValueIteration1(self):
        """
        terminal states have no reward, set val to 0
        self.values is a dictionary of {s : q_val, ....}
        I want to update the values associated with s

         """
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            new_values = self.values.copy()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    new_values[state] = max(
                        [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)],
                        default=0
                    )
            self.values = new_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).

        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          Q-Values: expected future utility from a q-state (chance node)
          Q*(s, a) = sum_s' of T(s, a, s') [R(s, a, s') + gamma V*(s')]
        """
        "*** YOUR CODE HERE ***"
        return sum(
            prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          This is policy Extraction
          V*(s) = max_a Q*(s, a)
          return the action with the highest Q-value
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        return max(
            self.mdp.getPossibleActions(state),
            key=lambda action: self.computeQValueFromValues(state, action),
            default=None
        )

    def getPolicy(self, state):
        return self.computeActionFromValues(state)


    def getAction(self, state):
        return self.computeActionFromValues(state)


    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = collections.defaultdict(set)
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        predecessors[nextState].add(state)

        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_qvalue = max(
                    (self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)),
                    default=0
                )
                diff = abs(max_qvalue - self.values[state])
                pq.update(state, -diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                    (self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)),
                    default=0
                )

            for p in predecessors[state]:
                max_qvalue = max(
                    (self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)),
                    default=0
                )
                diff = abs(max_qvalue - self.values[p])
                if diff > self.theta:
                    pq.update(p, -diff)