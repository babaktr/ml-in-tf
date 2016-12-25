import numpy as np

class GridWorld(object):

    def __init__(self, width, rand_seed=1):
        np.random.seed(rand_seed)
        self.width = np.max([3, width]) # Make sure game field isn't too small.
        self.reset()
        self.init_q_table()

    '''
    q_table holds the Q-values for n numbers of state-action pairs.
    Row is for state number
    Column is for actions 

    [[a_1, a_2, a_3, a_4],
     [a_1, a_2, a_3, a_4],
              ...
     [a_1, a_2, a_3, a_4]]
    '''
    def init_q_table(self):
        self.q_table = np.random.rand(self.width * self.width, 4)

    '''
    Reset environment.
    '''
    def reset(self):
        self.state = np.zeros((3, self.width, self.width))

        self.player = (0,0)
        self.pit = (0,0)
        self.goal = (0,0)

        self.init_grid()
        self.place_in_state()

        return self.state

    '''
    Count state number through player coordinates.
    '''
    def player_state_number(self, x, y):
        return (x+1) + (y*self.width) - 1

    '''
    Place player. pit and goal in a stacked matrix (state)
    '''    
    def place_in_state(self):
        player_pos = np.zeros((self.width, self.width))
        player_pos[self.player[1], self.player[0]] = 1

        pit_pos = np.zeros((self.width, self.width))
        pit_pos[self.pit[1], self.pit[0]] = 1

        goal_pos = np.zeros((self.width, self.width))
        goal_pos[self.goal[1], self.goal[0]] = 1

        self.state = np.zeros((3, self.width, self.width))

        self.state[0] = player_pos
        self.state[1] = pit_pos
        self.state[2] = goal_pos

    ''' 
    Initialize player in random location, but keep goal and pit stationary.
    '''
    def init_grid(self):
        # Sample from:
        arr = [0,0,0,0,1,2,3,3,3,3]
        #random_pair = np.random.choice(arr, size=2, replace=False)
        random_pair = (np.random.randint(0,self.width), np.random.randint(0, self.width))

        self.player = (random_pair[0], random_pair[1])
        self.pit = (1,1)
        self.goal = (self.width-2,self.width-2)

        self.place_in_state()

    ''' 
    Extract a state row of Q-values
    '''
    def q_values(self):
        q_table_row = self.q_table[self.player_state_number(self.player[0], self.player[1]),:]
        return q_table_row

    '''
    Perform action in environment
    '''
    def perform_action(self, action):
        # Save for later update
        self.old_player_state_number = self.player_state_number(self.player[0], self.player[1])
        old_loc = self.player

        # Up (row - 1)  
        if action == 0:
            if self.player[1] > 0:
                self.player = (self.player[0], self.player[1] - 1)
        # Down (row + 1)
        elif action == 1:
            if self.player[1] < 3:
                self.player = (self.player[0], self.player[1] + 1)
        # Left (column - 1)
        elif action == 2:
            if self.player[0] > 0:
                self.player = (self.player[0] - 1, self.player[1])
        # Right (column + 1)
        elif action == 3:
            if self.player[0] < 3:
                self.player = (self.player[0] + 1, self.player[1])


        if self.player == self.pit: # Player walked into pit, end episode
            reward = 0
            terminal = False
        elif self.player == self.goal: # Player walked in to goal, end episode
            reward = 1
            terminal = True
        elif old_loc == self.player: # Player did not move
            reward = -0.1
            terminal = False
        else: 
            reward = 0
            terminal = False

        self.place_in_state()

        return self.state, reward, terminal

    '''
    Update the table of Q-values
    '''
    def update_q_table(self, update, action, learning_rate, terminal):
        q_value = self.q_table[self.old_player_state_number, action]
        
        if terminal: 
            # Update with the given 'update' = r + gamma(maxQ(s',a'))
            self.q_table[self.old_player_state_number, action] = update
        else:
            # Update using the update rule Q(s,a) <- Q(s,a) + lr(r + gamma(maxQ(s',a')) - Q(s,a))
            self.q_table[self.old_player_state_number, action] = q_value + learning_rate * (update - q_value)

    '''
    Used to visualize a given state.
    '''
    def display_grid(self, state):
        grid = np.zeros((self.width,self.width), dtype='<U2')
        for i in range(0,self.width):
            for j in range(0,self.width):
                #grid[i,j] = ' '
                q_values = self.q_table[(i+1) + (j*4) - 1, :]
                string = ' '
                clar_str = str((i+1) + (j*4) - 1)
                qmax = np.argmax(q_values)
                #print qmax
                if qmax == 0:
                    string = '^'
                elif qmax == 1:
                    string = 'v'
                elif qmax == 2:
                    string = '<'
                elif qmax == 3:
                    string = '>'
                grid[j,i] = string
        
        grid[self.player[0], self.player[1]] = 'P' #player
        grid[self.goal[0], self.goal[1]] = '+' #goal
        grid[self.pit[0], self.pit[1]] = '-' #pit

        return grid
