"""
RL Framework
Author: Vikram Rao
AGV Environment
"""

import numpy as np
from Environment import *
import functools 

class AGV():
    """
    Simple AGV Environment
    ?? Expects specification (size, endpoints, barriers) to be given
    """

    MOVE_UP     = 0
    MOVE_DOWN   = 1
    MOVE_LEFT   = 2
    MOVE_RIGHT  = 3
    MOVE_PICK   = 4
    MOVE_DROP   = 5

    @staticmethod
    def state_idx( road_map, reward_success, holding, y, x, flipped ):
        """Compute the index of the state
        Holding - 1, 2, ... n_mach
        """
        n_mach = len(reward_success) + 1 # len number of items, plus none.
        y_max, x_max= road_map.shape

        K = max( [ n_mach, y_max, x_max ] )

        #if holding != 0 :
        #    holding -= 1

        idx = holding
        idx = idx*K + y
        idx = idx*K + x
        idx = idx*K + int(flipped)

        return idx
        # x + K*y + K^2*holding
        # raise NotImplemented()

    @staticmethod
    def idx_state( road_map, reward_success, state ):
        """Compute the state for the index"""
        n_mach = len(reward_success) + 1 # len number of items, plus none.
        y_max, x_max= road_map.shape

        K = max( [ n_mach, y_max, x_max ] )

        flipped, state = bool(state%K), state/K
        x, state = state%K, state/K
        y, state = state%K, state/K
        holding, state = state%K, state/K

        return holding, y, x, flipped
        #raise NotImplemented()

    @staticmethod
    def make_map_from_size( height, width ):
        raise NotImplemented()
        pass

    @staticmethod
    def make_map_from_file( fname ):
        spec = map( str.strip, open( fname ).readlines() )
        size = tuple( map( int, spec[0].split() ) )

        def row_to_int( row ):
            return map( int, row.split() )
        road_map = np.array( map( row_to_int, spec[ 1: ] ) )

        if size != road_map.shape:
            raise ValueError()

        g = np.max( road_map )
        n_mach = (g-1)/2

        reward_success = [ (m+1)*10 for m in range(n_mach) ]

        return road_map, reward_success

    @staticmethod
    def make_mdp( road_map, reward_success ):
        size = road_map.shape
        n_mach = len(reward_success)

        def make_map( road_map, holding, P ): 
            state_idx_ = functools.partial( AGV.state_idx, road_map, reward_success, holding )
            state_idx__ = functools.partial( AGV.state_idx, road_map, reward_success )

            def make_move( road_map, axis, y, x ):
                moves = []

                if axis == AGV.MOVE_UP:
                    moves.append( (state_idx_( y-1, x, False ), 1.0) )
                elif axis == AGV.MOVE_DOWN:
                    moves.append( (state_idx_( y+1, x, False ), 1.0) )
                elif axis == AGV.MOVE_LEFT:
                    moves.append( (state_idx_( y, x-1, False ), 1.0) )
                elif axis == AGV.MOVE_RIGHT:
                    moves.append( (state_idx_( y, x+1, False ), 1.0) )
                elif axis == AGV.MOVE_DROP:
                    moves.append( (state_idx__( 0, y, x, True ), 1.0) )
                elif axis == AGV.MOVE_PICK:
                    moves.append( (state_idx__( road_map[y][x]/2, y, x, True ), 1.0) )
                return moves

            # Up, down, left and right. Pick and drop.
            for y in xrange( size[ 0 ] ):
                for x in xrange( size[ 1 ] ):
                    for flipped in [True, False] :
                        s = state_idx_( y, x, flipped )
                        if y > 0 and road_map[ y-1, x ] != 1:
                            P[ AGV.MOVE_UP ][ s ] += make_move( road_map, AGV.MOVE_UP, y, x )
                        if y + 1 < size[0] and road_map[ y, x ] != 1:
                            P[ AGV.MOVE_DOWN ][ s ] += make_move( road_map, AGV.MOVE_DOWN, y, x )
                        if x > 0 and road_map[ y, x-1 ] != 1:
                            P[ AGV.MOVE_LEFT ][ s ] += make_move( road_map, AGV.MOVE_LEFT, y, x )
                        if x + 1 < size[1] and road_map[ y, x ] != 1:
                            P[ AGV.MOVE_RIGHT ][ s ] += make_move( road_map, AGV.MOVE_RIGHT, y, x )
                        if road_map[y][x] not in [0, 1] and road_map[y][x]%2 == 0 and holding == 0 :
                            #print 'Adding Pick', AGV.idx_state( road_map, reward_success, s )
                            P[ AGV.MOVE_PICK ][ s ] += make_move( road_map, AGV.MOVE_PICK, y, x )
                        if road_map[y][x] not in [0, 1] and road_map[y][x]%2 == 1 and 2*holding == road_map[y][x]-1 :
                            #print 'Adding Drop', AGV.idx_state( road_map, reward_success, s )
                            P[ AGV.MOVE_DROP ][ s ] += make_move( road_map, AGV.MOVE_DROP, y, x )
            return P

        # Create P, R
        K = max( size[0], size[1], n_mach )
        S = K*K*K*K # how many states?
        A = 6 # up down left right pick drop
        P = [ [ [] for i in xrange( S ) ] for j in xrange( A ) ]
        R = {}
        R_bias = 0# AGV.REWARD_BIAS
        start_set = []
        end_set = []

        P = make_map( road_map, 0, P )
        for i in range(1, 1+n_mach ):
            P = make_map( road_map, i, P )

        # Add rewards for dropping parts to the correct drop-point.
        for y in range( size[0] ):
            for x in range( size[1] ):
                if road_map[y][x] not in [0, 1] and road_map[y][x]%2 == 1:
                    # state_idx( road_map, reward_success, holding, y, x ):
                    item = (road_map[y][x] - 1)/2
                    s = AGV.state_idx( road_map, reward_success, item, y, x, False )
                    s_ = AGV.state_idx( road_map, reward_success, item, y, x, True )
                    s__ = AGV.state_idx( road_map, reward_success, 0, y, x, True )
                    R[ (s, s__) ] = reward_success[item-1]
                    R[ (s_, s__) ] = reward_success[item-1]
                    end_set.append(s__)

        for y in range( size[0] ):
            for x in range( size[1] ):
                start_set.append( AGV.state_idx( road_map, reward_success, 0, y, x, False ) )
                start_set.append( AGV.state_idx( road_map, reward_success, 0, y, x, True ) )

        #for e in end_set :
        #    print AGV.idx_state(road_map, reward_success, e)

        return S, A, P, R, R_bias, start_set, end_set

    @staticmethod
    def create( spec ):
        """Create a taxi from @spec"""
        if spec is None:
            road_map, reward_success = AGV.make_map_from_size( 5, 5 )
        else:
            road_map, reward_success = AGV.make_map_from_file( spec )
        return Environment( AGV, *AGV.make_mdp( road_map, reward_success ) )

    @staticmethod
    def reset_rewards( env, *args ):
        return env

