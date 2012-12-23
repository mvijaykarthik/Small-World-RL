"""
RL Framework
Authors: Arun Chaganty, Prateek Gaur
PlayRooms Environment
Room, with K arbitrary rewards distributed in the space.
"""

import numpy as np
from Environment import *
import functools 
import random
import itertools



class PlayRooms():
    """
    PlayRooms Environment
    Expects specification file to be given
    """

    WALL        = 1

    # Indices in the state tuple
    BALL_INDEX = 0
    BELL_INDEX = 1
    EYE_INDEX = 2
    HAND_INDEX = 3
    LIGHT_SWITCH_INDEX = 4
    MUSIC_ON_BLOCK_INDEX = 5
    MUSIC_OFF_BLOCK_INDEX = 6
    MARKER_INDEX = 7
    LIGHT_STATUS_INDEX = 8
    MUSIC_STATUS_INDEX = 9
    BELL_RING_INDEX = 10

    # Actions
    num_actions = 11
    MOVE_EYE_NORTH = 0
    MOVE_EYE_SOUTH = 1
    MOVE_EYE_EAST = 2
    MOVE_EYE_WEST = 3
    MOVE_EYE_TO_HAND = 4
    MOVE_EYE_TO_MARKER = 5
    MOVE_EYE_TO_RANDOM_OBJECT = 6
    MOVE_HAND_TO_EYE = 7
    MOVE_MARKER_TO_EYE = 8
    INTERRACT = 9
    MOVE_BLOCK = 10
    # # North South East West movements have accuracies
    # ACCURACY = 0.67
    
    REWARD_BIAS = -1
    REWARD_FAILURE = -20 - REWARD_BIAS
    REWARD_SUCCESS = 50 - REWARD_BIAS
    REWARD_SUCCESS_VAR = 1
    REWARD_CHECKPOINT = 0 # - REWARD_BIAS

    class RewardPlayRoom(dict):
        # If light is off, music is on, and bell rings
        # Then reward is non zero
        def __init__(self, *args):
            # Last element is room_map
            dict.__init__(self, args[:-1])
            self.room_map = args[-1]
     
        def __getitem__(self, key):
            dest = key[1]
            state = PlayRooms.idx_state(self.room_map, dest)
            if state[PlayRooms.LIGHT_STATUS_INDEX] == 0 and \
                    state[PlayRooms.MUSIC_STATUS_INDEX] != 0 and \
                    state[PlayRooms.BELL_RING_INDEX] != 0:
                # its the goal
                reward = np.random.normal( PlayRooms.REWARD_SUCCESS - PlayRooms.REWARD_BIAS, PlayRooms.REWARD_SUCCESS_VAR)
                return reward
            return 0
        
    
    @staticmethod
    def encode_pos(pos, offset, num_cols, area, st):
        """
        Function used in computing index of a state.
        """
        if type(pos)!=int and type(pos) != long:
            return offset*area, offset * (pos[1] + pos[0]*num_cols) + st
        else:
            return offset*area, offset * (pos) + st
        
    @staticmethod
    def state_idx( room_map, ball_pos, bell_pos, eye_pos, hand_pos,
                   light_switch_pos, music_on_block_pos, music_off_block_pos, marker_pos, light_status,
                   music_status, bell_ring):
        """Compute the index of the state"""

        size = room_map.shape
        num_rows = size[0]
        num_cols = size[1]
        area = num_rows*num_cols
        
        # st, offset = x, size[1]
        # st, offset = st + offset * y, offset * size[0]
        st = 0
        offset = 1
        offset, st =  PlayRooms.encode_pos(ball_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(bell_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(eye_pos, offset, num_cols, area, st)        
        offset, st =  PlayRooms.encode_pos(hand_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(light_switch_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(music_on_block_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(music_off_block_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(marker_pos, offset, num_cols, area, st)
        offset, st =  PlayRooms.encode_pos(light_status, offset, num_cols, 2, st)
        offset, st =  PlayRooms.encode_pos(music_status, offset, num_cols, 2, st)
        offset, st =  PlayRooms.encode_pos(bell_ring, offset, num_cols, 2, st)        
        return st
    
    @staticmethod
    def decode_pos(num_cols, area, state, isPos = True):
        """
        Function used in decoding the state from index
        """
        bits = state % area
        if isPos:
            pos =  [ bits / num_cols, bits % num_cols ]
            return state / area, pos
        else:
            # Bool
            pos =   bits % num_cols
            return state / area, pos
        
    @staticmethod
    def idx_state( room_map, state ):
        size = room_map.shape
        num_rows = size[0]
        num_cols = size[1]
        area = num_rows*num_cols
        """Compute the state for the index"""
        
        state, ball_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, bell_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, eye_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, hand_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, light_switch_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)   
        state, music_on_block_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, music_off_block_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, marker_pos = PlayRooms.decode_pos(num_cols, area, state, isPos = True)
        state, light_status = PlayRooms.decode_pos(num_cols, 2, state, isPos = False)
        state, music_status = PlayRooms.decode_pos(num_cols, 2, state, isPos = False)
        state, bell_ring = PlayRooms.decode_pos(num_cols, 2, state, isPos = False)                
        return [ball_pos, bell_pos, eye_pos, hand_pos, light_switch_pos, music_on_block_pos, music_off_block_pos, marker_pos, light_status, music_status, bell_ring]
    
        
    @staticmethod
    def make_map_from_size( height, width ):
        raise NotImplemented()
        pass

    @staticmethod
    def make_map_from_txt_file( fname ):
        spec = map( str.strip, open( fname ).readlines() )
        size = tuple( map( int, spec[0].split() ) )

        def row_to_int( row ):
            return map( int, row.split() )
        room_map = np.array( map( row_to_int, spec[ 1: ] ) )

        if size != room_map.shape:
            raise ValueError()

        return room_map

    @staticmethod
    def make_map_from_tsv_file( fname ):
        spec = open( fname ).readlines()
        width = len(spec[0].split('\t'))
        height = len(spec)
        size = (height, width)

        def row_to_int( row ):
            row = row.split('\t')
            for i in xrange(len(row)):
                if row[i] == 'F': 
                    row[i] = 0
                else:
                    row[i] = 1
            return row
        room_map = np.array( map( row_to_int, spec ) )

        if size != room_map.shape:
            raise ValueError()

        return room_map

    
    @staticmethod
    def get_all_points(room_map):
        """
        Returns all valid points for the eye in playroom domain.
        """
        points = []
        size = room_map.shape
        num_rows = size[0]
        num_cols = size[1]
        
        for i in range(0, num_rows):
            for j in range(0, num_cols):
                if room_map[i][j] != PlayRooms.WALL:
                    points.append((i,j))
        return points

    @staticmethod
    def get_random_start_state( room_map ):
        """
        Generates random distinct positions for the objects in the playroom
        """
        all_pts = PlayRooms.get_all_points(room_map)
        selected_pts = random.sample(all_pts, 8)
        st = PlayRooms.state_idx( room_map, *selected_pts, light_status = 0, bell_ring = 0, music_status = 0 )
        return st
    
    @staticmethod
    def make_mdp( room_map, K ):
        size = room_map.shape
        min_size = len( room_map[ room_map == 0] )

        state_idx = PlayRooms.state_idx
        idx_state = PlayRooms.idx_state

        area = size[0]* size[1]
        S = (area ** 8) * (2**3)
        A = PlayRooms.num_actions
        
        # Support large argument for range
        xrange = lambda stop: iter(itertools.count().next, stop)
        
        R = {}
        R_bias = PlayRooms.REWARD_BIAS

        # Populate the P table
        # Class behaving like array for P[action][state]
        class LookUpP1:
            def __init__(self,room_map):
                self.room_map = room_map
            def __getitem__(self, index):
                lookup = LookUpP2(index, room_map)
                return lookup
        
        class LookUpP2:
            def __init__(self, action, room_map):
                self.action = action
                self.room_map = room_map
            def __getitem__(self, state):
                state = idx_state(self.room_map, state)
                state[PlayRooms.BELL_RING_INDEX] = 0
                size = self.room_map.shape
                
                def equal(pos1, pos2):
                    return pos1[0] == pos2[0] and pos1[1] == pos2[1]
                
                if self.action == PlayRooms.MOVE_EYE_NORTH:
                    eyepos = state[PlayRooms.EYE_INDEX]
                    if eyepos[0] > 0 and self.room_map[eyepos[0] - 1, eyepos[1]] & PlayRooms.WALL == 0 :
                        eyepos[0] -= 1
                    return [(state_idx(self.room_map, *state),1)]
                
                elif self.action == PlayRooms.MOVE_EYE_SOUTH:
                    eyepos = state[PlayRooms.EYE_INDEX]

                    if eyepos[0] + 1 < size[0] and self.room_map[eyepos[0] + 1, eyepos[1]] & PlayRooms.WALL == 0 :
                        eyepos[0] += 1
                    return [(state_idx(self.room_map, *state),1)]                
         
         
                elif self.action == PlayRooms.MOVE_EYE_EAST:
                    eyepos = state[PlayRooms.EYE_INDEX]
                    if eyepos[1] + 1 < size[1] and self.room_map[eyepos[0], eyepos[1] + 1] & PlayRooms.WALL == 0 :
                        eyepos[1] += 1
                    return [(state_idx(self.room_map, *state),1)]                                
         
                elif self.action == PlayRooms.MOVE_EYE_WEST:
                    eyepos = state[PlayRooms.EYE_INDEX]
                    if eyepos[1]  > 0 and self.room_map[eyepos[0], eyepos[1] - 1] & PlayRooms.WALL == 0 :
                        eyepos[1] -= 1
                    return [(state_idx(self.room_map, *state),1)]                                
         
                elif self.action == PlayRooms.MOVE_EYE_TO_HAND:
                    hand_pos = state[PlayRooms.HAND_INDEX]
                    state[PlayRooms.EYE_INDEX] = hand_pos[:]
                    return [(state_idx(self.room_map, *state),1)]                            
         
                elif self.action == PlayRooms.MOVE_EYE_TO_MARKER:
                    marker_pos = state[PlayRooms.MARKER_INDEX]
                    state[PlayRooms.EYE_INDEX] = marker_pos[:]
                    return [(state_idx(self.room_map, *state),1)]                            
         
                elif self.action == PlayRooms.MOVE_EYE_TO_RANDOM_OBJECT:
                    choices = [PlayRooms.BALL_INDEX, PlayRooms.BELL_INDEX, PlayRooms.HAND_INDEX, PlayRooms.MUSIC_ON_BLOCK_INDEX, PlayRooms.MUSIC_OFF_BLOCK_INDEX, PlayRooms.MARKER_INDEX]
                    choice = random.choice(choices)
                    possible_future_states = []
                    prob = 1.0/len(choices)            
                    for choice in choices:
                        newState = []
                        for item in state:
                            newState.append(item)
                        newState[PlayRooms.EYE_INDEX] = state[choice][:]
                        possible_future_states.append((state_idx(self.room_map, *newState), prob))
                    return possible_future_states
         
                elif self.action == PlayRooms.MOVE_HAND_TO_EYE:
                    eye_pos = state[PlayRooms.EYE_INDEX]
                    state[PlayRooms.HAND_INDEX] = eye_pos[:]
                    return [(state_idx(self.room_map, *state),1)]                            
         
                elif self.action == PlayRooms.MOVE_MARKER_TO_EYE:
                    eye_pos = state[PlayRooms.EYE_INDEX]
                    state[PlayRooms.MARKER_INDEX] = eye_pos[:]
                    return [(state_idx(self.room_map, *state),1)]
                
                
                # Move block to random adjacent position
                elif self.action == PlayRooms.MOVE_BLOCK:
                    eye_pos = state[PlayRooms.EYE_INDEX]
                    hand_pos = state[PlayRooms.HAND_INDEX]
                    ball_pos = state[PlayRooms.BALL_INDEX]
                    bell_pos = state[PlayRooms.BELL_INDEX]
                    light_switch_pos = state[PlayRooms.LIGHT_SWITCH_INDEX]
                    music_on_pos = state[PlayRooms.MUSIC_ON_BLOCK_INDEX]
                    music_off_pos = state[PlayRooms.MUSIC_OFF_BLOCK_INDEX]
                    marker_pos = state[PlayRooms.MARKER_INDEX]
                    if not equal(eye_pos, hand_pos):
                        return []
                    elif equal(eye_pos, music_on_pos):
                        possible_future_states = []
                        prob = 1.0/4
                        for xy in [0,1]:
                            for perturb in [-1,1]:
                                newState = []
                                for item in state:
                                    newState.append(item)
                                newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] += perturb

                                # Fix if position is out of playroom
                                if newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] < 0 :
                                    newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] = 0
                                if xy == 0:
                                    if newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] >= size[0]:
                                        newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] = size[0]
                                if xy == 1:
                                    if newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] >= size[1]:
                                        newState[PlayRooms.MUSIC_ON_BLOCK_INDEX][xy] = size[1]                                
                                        
                                possible_future_states.append((state_idx(self.room_map, *newState), prob))
                        return possible_future_states
         
                    elif equal(eye_pos, music_off_pos):
                        possible_future_states = []
                        prob = 1.0/4
                        for xy in [0,1]:
                            for perturb in [-1,1]:
                                newState = []
                                for item in state:
                                    newState.append(item)
                                newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] += perturb

                                # Fix if position is out of playroom
                                if newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] < 0 :
                                    newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] = 0
                                if xy == 0:
                                    if newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] >= size[0]:
                                        newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] = size[0]
                                if xy == 1:
                                    if newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] >= size[1]:
                                        newState[PlayRooms.MUSIC_OFF_BLOCK_INDEX][xy] = size[1]                                
                                
                                possible_future_states.append((state_idx(self.room_map, *newState), prob))
                        return possible_future_states
                    
                    else:
                        return []
                # Interract with object
                # Music on block => Music status becomes on
                # Music off block => Music status becomes off
                # ball => ball is kicked to marker, and if it
                #    hits the bell, then bell moves to random adjacent block and bell_ring is on
                # light_switch => toggles light status
                elif self.action == PlayRooms.INTERRACT:
                    eye_pos = state[PlayRooms.EYE_INDEX]
                    hand_pos = state[PlayRooms.HAND_INDEX]
                    ball_pos = state[PlayRooms.BALL_INDEX]
                    bell_pos = state[PlayRooms.BELL_INDEX]
                    light_switch_pos = state[PlayRooms.LIGHT_SWITCH_INDEX]
                    music_on_pos = state[PlayRooms.MUSIC_ON_BLOCK_INDEX]
                    music_off_pos = state[PlayRooms.MUSIC_OFF_BLOCK_INDEX]
                    marker_pos = state[PlayRooms.MARKER_INDEX]
                    
         
                    if not equal(eye_pos, hand_pos):
                        return []

                    # Kick ball
                    elif equal(eye_pos, ball_pos):
                        state[PlayRooms.BALL_INDEX] = marker_pos[:]
                        
                        # Check if it hits bell. Bell rings if ball hits bell
                        if equal(marker_pos, bell_pos):
                            state[PlayRooms.BELL_RING_INDEX] = 1
                            possible_future_states = []
                            prob = 1.0/4
                            for xy in [0,1]:
                                for perturb in [-1,1]:
                                    newState = []
                                    for item in state:
                                        newState.append(item)
                                    newState[PlayRooms.BELL_INDEX][xy] += perturb
                                    # Fix if position is out of playroom
                                    if newState[PlayRooms.BELL_INDEX][xy] < 0 :
                                        newState[PlayRooms.BELL_INDEX][xy] = 0
                                    if xy == 0:
                                        if newState[PlayRooms.BELL_INDEX][xy] >= size[0]:
                                            newState[PlayRooms.BELL_INDEX][xy] = size[0]
                                    if xy == 1:
                                        if newState[PlayRooms.BELL_INDEX][xy] >= size[1]:
                                            newState[PlayRooms.BELL_INDEX][xy] = size[1]                                
                                        
                                    possible_future_states.append((state_idx(self.room_map, *newState), prob))
                            return possible_future_states
                            
                        return [(state_idx(self.room_map, *state),1)]

                    # Toggle light
                    elif equal(eye_pos, light_switch_pos):
                        state[PlayRooms.LIGHT_STATUS_INDEX] = 1 - state[PlayRooms.LIGHT_STATUS_INDEX]
                        return [(state_idx(self.room_map, *state),1)]                

                    # Music On
                    elif equal(eye_pos, music_on_pos):
                        # Light should be on to know color
                        if not state[PlayRooms.LIGHT_STATUS_INDEX]:
                            return []
                        state[PlayRooms.MUSIC_STATUS_INDEX] = 1
                        return [(state_idx(self.room_map, *state),1)]                

                    # Music off
                    elif equal(eye_pos, music_off_pos):
                        if not state[PlayRooms.LIGHT_STATUS_INDEX]:
                            return []
                        state[PlayRooms.MUSIC_STATUS_INDEX] = 0
                        return [(state_idx(self.room_map, *state),1)]
                    else:
                        return []
                else:
                    return []
                    

        P = LookUpP1(room_map)
        # Add rewards to all states that transit into the goal state
        # Generate Start Set        
        start_set = [PlayRooms.get_random_start_state(room_map)]
        end_set = None
        
        R = PlayRooms.RewardPlayRoom(room_map)
                    
        return S, A, P, R, R_bias, start_set, None, PlayRooms.end_condition, room_map



    @staticmethod
    def end_condition(stateID, room_map):
        # End condition. Light is off, Music is on, and Bell rings
        state = PlayRooms.idx_state(room_map, stateID)
        if state[PlayRooms.LIGHT_STATUS_INDEX] == 0 and \
                state[PlayRooms.MUSIC_STATUS_INDEX] != 0 and \
                state[PlayRooms.BELL_RING_INDEX] != 0:
            return True
        else:
            return False
     
    @staticmethod
    def create( spec, K=1 ):
        """Create a room from @spec"""
        if spec is None:
            raise NotImplemented
        else:
            extn = spec.split('.')[-1]
            if extn == "tsv":
                room_map = PlayRooms.make_map_from_tsv_file( spec )
            else:
                room_map = PlayRooms.make_map_from_txt_file( spec )

        return Environment( PlayRooms, *PlayRooms.make_mdp( room_map, K ) )

    @staticmethod
    def reset_rewards( env, spec, K=1 ):
        if spec is None:
            raise NotImplemented
        else:
            extn = spec.split('.')[-1]
            if extn == "tsv":
                room_map = PlayRooms.make_map_from_tsv_file( spec )
            else:
                room_map = PlayRooms.make_map_from_txt_file( spec )

        state_idx = PlayRooms.state_idx
        idx_state = PlayRooms.idx_state

        start_set = [PlayRooms.get_random_start_state(room_map)]        
        end_set = []

        R = PlayRooms.RewardPlayRoom(room_map)

        return Environment(PlayRooms, env.S, env.A, env.P, R, env.R_bias, start_set, None, PlayRooms.end_condition, env.room_map)

        # return Environment( PlayRooms, env.S, env.A, env.P, R, env.R_bias, start_set, end_set )

