import numpy as np

class QTable(object):
    '''
    q_table holds the Q-values for n numbers of state-action pairs.
    Row is for state number
    Column is for actions 

    [[a_1, a_2, a_3, a_4],
     [a_1, a_2, a_3, a_4],
              ...
     [a_1, a_2, a_3, a_4]]
    '''
    def __init__(self, field_size, random_seed):
        np.random.seed(random_seed)
        self.q_table = np.random.rand(field_size * field_size, 4)

    ''' 
    Extract a state row of Q-values.
    '''
    def get_state_q(self, state_row):
        q_table_row = self.q_table[state_row,:]
        return q_table_row
        
    '''
    Update the table of Q-values
    '''
    def update_state_q(self, update, state_row, action, learning_rate, terminal):
        q_value = self.q_table[state_row, action]
        
        if terminal: 
            # Update with the given 'update' = r + gamma(maxQ(s',a'))
            self.q_table[state_row, action] = update
        else:
            # Update using the update rule Q(s,a) <- Q(s,a) + lr(r + gamma(maxQ(s',a')) - Q(s,a))
            self.q_table[state_row, action] = q_value + learning_rate * (update - q_value)