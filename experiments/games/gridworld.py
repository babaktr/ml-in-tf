import numpy as np

class GridWorld(object):

    def __init__(self, field_size, rand_seed=1):
        np.random.seed(rand_seed)
        self.field_size = np.max([3, field_size]) # Sets a minimal field_size of 3.
        self.reset()

    '''
    Reset field.
    '''
    def reset(self):
        self.state = np.zeros((3, self.field_size, self.field_size))

        self.actor = (0,0)
        self.pit = (0,0)
        self.goal = (0,0)

        self.init_grid()
        self.place_in_state()

        return self.state

    '''
    Count state number through actor coordinates.
    '''
    def actor_state_row(self):
        return (self.actor[0]+1) + (self.actor[1]*self.field_size) - 1

    '''
    Place player, pit and goal each in a stacked matrix (state).

    E.g:
    Actor - a
    Pit - p
    goal - g
    [[[a_11, a_12, a_13],
      [a_21, a_22, a_23],
      [a_31, a_32, a_33]],
     [[p_11, p_12, p_13],
      [p_21, p_22, p_23],
      [p_31, p_32, p_33]],
     [[g_11, g_12, g_13],
      [g_21, g_22, g_23],
      [g_31, g_32, g_33]]]
    '''    
    def place_in_state(self):
        actor_pos = np.zeros((self.field_size, self.field_size))
        actor_pos[self.actor[1], self.actor[0]] = 1

        pit_pos = np.zeros((self.field_size, self.field_size))
        pit_pos[self.pit[1], self.pit[0]] = 1

        goal_pos = np.zeros((self.field_size, self.field_size))
        goal_pos[self.goal[1], self.goal[0]] = 1

        self.state = np.zeros((3, self.field_size, self.field_size))

        self.state[0] = actor_pos
        self.state[1] = pit_pos
        self.state[2] = goal_pos

    def compress_state(self):
        state = np.zeros((self.field_size, self.field_size))
        state[self.actor[1], self.actor[0]] = 1

        state[self.pit[1], self.pit[0]] = 0.7

        state[self.goal[1], self.goal[0]] = 0.3

        return state

    ''' 
    Initialize actor in random location, but keep goal and pit stationary.
    '''
    def init_grid(self):
        # Sample from:
        arr = [0,0,0,0,1,2,3,3,3,3]
        random_pair = (np.random.randint(0,self.field_size), np.random.randint(0, self.field_size))

        self.actor = (random_pair[0], random_pair[1])
        self.pit = (1,1)
        self.goal = (self.field_size-2,self.field_size-2)

        self.place_in_state()

    '''
    Perform action in field.
    '''
    def perform_action(self, action):
        # Save for later update
        self.old_actor_state_row = self.actor_state_row()
        old_loc = self.actor

        # Up (row - 1)  
        if action == 0:
            if self.actor[1] > 0:
                self.actor = (self.actor[0], self.actor[1] - 1)
        # Down (row + 1)
        elif action == 1:
            if self.actor[1] < self.field_size - 1:
                self.actor = (self.actor[0], self.actor[1] + 1)
        # Left (column - 1)
        elif action == 2:
            if self.actor[0] > 0:
                self.actor = (self.actor[0] - 1, self.actor[1])
        # Right (column + 1)
        elif action == 3:
            if self.actor[0] < self.field_size - 1:
                self.actor = (self.actor[0] + 1, self.actor[1])


        if self.actor == self.pit: # Actor walked into pit, end episode
            reward = 0
            terminal = False
        elif self.actor == self.goal: # Actor walked in to goal, end episode
            reward = 1
            terminal = True
        elif old_loc == self.actor: # Actor did not move
            reward = -0.1
            terminal = False
        else: 
            reward = 0
            terminal = False

        self.place_in_state()

        return self.state, reward, terminal

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
        
        grid[self.actor[0], self.actor[1]] = 'P' #actor
        grid[self.goal[0], self.goal[1]] = '+' #goal
        grid[self.pit[0], self.pit[1]] = '-' #pit

        return grid
