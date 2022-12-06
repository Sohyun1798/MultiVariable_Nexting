import numpy as np

NUM_STATES = 19
START = 9
END_0 = 0
END_1 = 20

class TDlambda:
    def __init__(self, alpha=0.1, gamma=0.9, lmbda=0.8):
        self.weights = np.zeros(NUM_STATES+2)
        self.z = np.zeros(NUM_STATES+2)
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        
    def value(self, state):
        v = self.weights[state]
        return v
    
    def updateZ(self, state):
        dev = 1
        self.z *= self.gamma*self.lmbda
        self.z[state] += dev
    
    def learn(self, state, nxtState, reward):
        delta = reward + self.gamma*self.value(nxtState) - self.value(state)
        delta *= self.alpha
        self.weights += delta*self.z

    def chooseAction(self): #?
        action = np.random.choice(self.actions)
        return action 
    
    def takeAction(self, action):
        new_state = self.state
        if not self.end:
            if action == "left":
                new_state = self.state-1
            else:
                new_state = self.state+1

            if new_state in [END_0, END_1]:
                self.end = True
        return new_state

    def giveReward(self, state):
        if state == END_0:
            return -1
        if state == END_1:
            return 1
        # other states
        return 0

    def play(self, valueFunc, rounds=100):
        for _ in range(rounds):
            self.reset()      
            action = self.chooseAction()
            while not self.end:
                nxtState = self.takeAction(action)  # next state
                self.reward = self.giveReward(nxtState)  # next state-reward

                valueFunc.updateZ(self.state)
                valueFunc.learn(self.state, nxtState, self.reward)
                
                self.state = nxtState
                action = self.chooseAction()
            if self.debug:
                print("end at {} reward {}".format(self.state, self.reward))