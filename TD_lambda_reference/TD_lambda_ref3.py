NUM_STATES = 1000
START = 500
END_0 = 0
END_1 = 1001

TRUE_VALUES = np.arange(-1001, 1003, 2) / 1001.0

class RandomWalk:

    def __init__(self, step=1, lr=2e-5, exp_rate=0.2, gamma=1, debug=True):
        self.state = START
        self.actions = ["left", "right"]
        self.end = False
        self.n = step
        self.lr = lr
        self.gamma = gamma
        self.exp_rate = exp_rate
        self.debug = debug

    def chooseAction(self, valueFunc):
        # choose an action based on the current state, 
        if np.random.uniform(0, 1) <= self.exp_rate:
            # random action
            return np.random.choice(self.actions)
        else:
            # greedy action
            values = {}
            for a in self.actions:
                value = valueFunc.value([self.state], a)
                values[a] = value
            return np.random.choice([k for k, v in values.items() if v==max(values.values())])

        return action

    def takeAction(self, action):
        # choose steps from 1 to 100
        steps = np.random.choice(range(1, 101))
        if action == "left":
            state = self.state - steps
        else:
            state = self.state + steps
        # judge if end of game
        if state <= END_0 or state >= END_1:
            self.end = True
            if state <= END_0:
                state = END_0
            else:
                state = END_1

        self.state = state
        return state

    def giveReward(self):
        if self.state == END_0:
            return -1
        if self.state == END_1:
            return 1
        return 0

    def reset(self):
        self.state = START
        self.end = False

    def play(self, valueFunction, rounds=1e5):
        for rnd in range(1, rounds+1):
            self.reset()
            t = 0
            T = np.inf
            action = self.chooseAction(valueFunction)

            actions = [action]
            states = [self.state]
            rewards = [0]
            while True:
                if t < T:
                    state = self.takeAction(action)  # next state
                    reward = self.giveReward()  # next state-reward

                    states.append(state)
                    rewards.append(reward)

                    if self.end:
                        if self.debug:
                            if rnd % 100 == 0:
                                print("Round {}: End at state {} | number of states {}".format(rnd, state,
                                                                                               len(states)))
                        T = t + 1
                    else:
                        action = self.chooseAction(valueFunction)
                        actions.append(action)  # next action
                # state tau being updated
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                        G += np.power(self.gamma, i - tau - 1) * rewards[i]
                    if tau + self.n < T:
                        state = states[tau + self.n]
                        G += np.power(self.gamma, self.n) * valueFunction.value([state], actions[tau+self.n])
                    # update value function
                    state = states[tau]
                    action = actions[tau]
                    valueFunction.update([state], action, G)

                if tau == T - 1:
                    break

                t += 1

feature_ranges = [[0, 1000]]  # 1 features
number_tilings = 50
bins = [[10] for _ in range(number_tilings)]
offsets = [[i] for i in np.linspace(0, 1000, number_tilings)]

tilings = create_tilings(feature_ranges=feature_ranges, number_tilings=number_tilings, bins=bins, offsets=offsets)

rw = RandomWalk()
valueFunc = QValueFunction(tilings, ["left", "right"], 0.01)

rw.play(valueFunction=valueFunc, rounds=1000)