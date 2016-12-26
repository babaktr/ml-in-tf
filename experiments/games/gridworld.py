import numpy as np

class GridWorld(object):

    def __init__(self, field_size, rand_seed=1):
        np.random.seed(rand_seed)
        self.field_size = np.max([3, field_size]) # Sets a minimal field_size of 3.
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
        self.q_table = np.random.rand(self.field_size * self.field_size, 4)

    '''
    Reset field.
    '''
    def reset(self):
        self.state = np.zeros((3, self.field_size, self.field_size))

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
        return (x+1) + (y*self.field_size) - 1

    '''
    Place player, pit and goal each in a stacked matrix (state).

    E.g:
    Player - pl
    Pit - p
    goal - g

    [[[pl_11, pl_12, pl_13],
      [pl_21, pl_22, pl_23],
      [pl_31, pl_32, pl_33]],
     [[p_11, p_12, p_13],
      [p_21, p_22, p_23],
      [p_31, p_32, p_33]],
     [[g_11, g_12, g_13],
      [g_21, g_22, g_23],
      [g_31, g_32, g_33]]]
    '''    
    def place_in_state(self):
        player_pos = np.zeros((self.field_size, self.field_size))
        player_pos[self.player[1], self.player[0]] = 1

        pit_pos = np.zeros((self.field_size, self.field_size))
        pit_pos[self.pit[1], self.pit[0]] = 1

        goal_pos = np.zeros((self.field_size, self.field_size))
        goal_pos[self.goal[1], self.goal[0]] = 1

        self.state = np.zeros((3, self.field_size, self.field_size))

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
        random_pair = (np.random.randint(0,self.field_size), np.random.randint(0, self.field_size))

        self.player = (random_pair[0], random_pair[1])
        self.pit = (1,1)
        self.goal = (self.field_size-2,self.field_size-2)

        self.place_in_state()

    ''' 
    Extract a state row of Q-values.
    '''
    def q_values(self):
        q_table_row = self.q_table[self.player_state_number(self.player[0], self.player[1]),:]
        return q_table_row

    '''
    Perform action in field.
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
            if self.player[1] < self.field_size - 1:
                self.player = (self.player[0], self.player[1] + 1)
        # Left (column - 1)
        elif action == 2:
            if self.player[0] > 0:
                self.player = (self.player[0] - 1, self.player[1])
        # Right (column + 1)
        elif action == 3:
            if self.player[0] < self.field_size - 1:
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
        grid = np.zeros((self.field_size,self.field_size), dtype='<U2')
        for i in range(0,self.field_size):
            for j in range(0,self.field_size):
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
