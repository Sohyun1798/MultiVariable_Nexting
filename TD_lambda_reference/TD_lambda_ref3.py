import numpy as np

class MountainCar:
    
    def __init__(self, n=1, exp_rate=0.1, gamma=1, debug=True):
        self.actions = [-1, 0, 1]  # reverse, 0 and forward throttle
        self.state = (-0.5, 0)  # position, velocity
        self.exp_rate = exp_rate
        self.gamma = 1
        self.end = False
        self.n = n  # step of learning
        self.debug = debug
        
    def reset(self):
        pos = np.random.uniform(-0.6, -0.4)
        self.end = False
        self.state = (pos, 0)
        
    def takeAction(self, action):
        pos, vel = self.state
        
        vel_new = vel + 0.001*action - 0.0025*np.cos(3*pos)
        vel_new = min(max(vel_new, VELOCITY_BOUND[0]), VELOCITY_BOUND[1])
        
        pos_new = pos + vel_new
        pos_new = min(max(pos_new, POSITION_BOUND[0]), POSITION_BOUND[1])
        
        if pos_new == POSITION_BOUND[0]:
            # reach leftmost, set speed to 0
            vel_new = 0
        self.state = (pos_new, vel_new)
        return self.state
    
    def chooseAction(self, valueFunc):
        # choose an action based on the current state, 
        if np.random.uniform(0, 1) <= self.exp_rate:
            # random action
            return np.random.choice(self.actions)
        else:
            # greedy action
            values = {}
            for a in self.actions:
                value = valueFunc.value(self.state, a)
                values[a] = value
            return np.random.choice([k for k, v in values.items() if v==max(values.values())])
        
    def giveReward(self):
        pos, _ = self.state
        if pos == POSITION_BOUND[1]:
            self.end = True
            return 0
        return -1
        
    def play(self, valueFunction, rounds=1):
        for rnd in range(1, rounds+1):
            self.reset()
            t = 0
            T = np.inf
            action = self.chooseAction(valueFunction)
            
            actions = [action]
            states = [self.state]
            rewards = [-1]
            
            while True:
                if t < T:
                    state = self.takeAction(action)  # next state
                    reward = self.giveReward()  # next state-reward
                    
                    states.append(state)
                    rewards.append(reward)
                    
                    if self.end:
                        if self.debug:
                            if rnd % 100 == 0:
                                print("Round {}: End at state {} | number of states {}".format(rnd, state, len(states)))
                        T = t+1
                    else:
                        action = self.chooseAction(valueFunction)
                        actions.append(action)  # next action
                # state tau being updated
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau+1, min(tau+self.n+1, T+1)):
                        G += np.power(self.gamma, i-tau-1)*rewards[i]
                    if tau+self.n < T:
                        state = states[tau+self.n]
                        G += np.power(self.gamma, self.n)*valueFunction.value(state, actions[tau+self.n])
                    # update value function
                    state = states[tau]  # tau is the state to update
                    valueFunction.update(state, actions[tau], G)
                    
                if tau == T-1:
                    break
                
                t += 1

VELOCITY_BOUND = [-0.07, 0.07]
POSITION_BOUND = [-1.2, 0.5]
ACTIONS = [-1, 0, 1]

feature_ranges = [POSITION_BOUND, VELOCITY_BOUND]  # 2 features
number_tilings = 8
bins = [[16, 16] for _ in range(number_tilings)]
offsets = [[i, j] for i, j in zip(np.linspace(POSITION_BOUND[0], POSITION_BOUND[1], number_tilings), np.linspace(VELOCITY_BOUND[0], VELOCITY_BOUND[1], number_tilings))]

tilings = create_tilings(feature_ranges=feature_ranges, number_tilings=number_tilings, bins=bins, offsets=offsets)

print("tiling shape: \n", tilings.shape)
print("offsets: \n", offsets)

valueFunc = QValueFunction(tilings, ACTIONS, 0.3)
mc = MountainCar()  # 1 step sarsa
mc.play(valueFunc, rounds=1)
