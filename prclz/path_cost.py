from abc import ABC, abstractmethod
import i_topology
import igraph
import numpy as np 
'''
NOTE: 
This file defines an API for flexibly calculating the weight/cost of 
a given path. Because there are myriad ways to operationalize given
characteristics, to create a new one, just subclass the BasePathCost
and define a cost_of_path method
'''

class BasePathCost(ABC):

    def __init__(self,
                 lambda_dist=1.0,
                 lambda_turn_angle=0.0,
                 lambda_degree=0.0,
                 lambda_width=0.0):
        self.lambda_dist = lambda_dist
        self.lambda_turn_angle = lambda_turn_angle
        self.lambda_degree = lambda_degree
        self.lambda_width = lambda_width

    @abstractmethod
    def __call__(self, 
                 g: i_topology.PlanarGraph,
                 path_seq: igraph.EdgeSeq,
                 return_compnents = False ) -> float:
        raise NotImplementedError("Subclass of BasePathCost must implement specific method for calculating the path cost")


class FlexCost(BasePathCost):

    '''
    NOTE: this needs to be fixed to not double
    count, now that we are calculating the marginal
    cost of the path, rather than the full path cost
    '''

    def __call__(self, 
                 g: i_topology.PlanarGraph, 
                 path_seq: igraph.EdgeSeq,
                 return_compnents=False) -> float:
        
        path_cost = 0
        path_cost_components = {}
        path_length = len(path_seq)

        i = 0
        for p1, p2 in zip(path_seq[0:-1], path_seq[1:]):
            v1 = g.vs.select(name=p1)[0]
            v2 = g.vs.select(name=p2)[0]

            if self.lambda_dist > 0:
                # Euclidean distance
                distance = g.es.select(_between=([v1.index], [v2.index]))['eucl_dist'][0]
                dist_cost = self.lambda_dist * distance

                # Divided by width, so width mitigates long distance
                if self.lambda_width:
                    width = g.es.select(_between=([v1.index], [v2.index]))['width'][0]
                    width_scalar = 1 / (self.lambda_width * width)
                else:
                    width_scalar = 1
                path_cost += (dist_cost * width_scalar)
                path_cost_components['dist'] = path_cost_components.get('dist', 0) + dist_cost
                path_cost_components['width'] = path_cost_components.get('width', 0) + width_scalar

            if self.lambda_degree > 0:
                # Deviation from 4-way intersection
                dev_from_4 = np.abs(v2.degree() - 4)
                if i == 0:
                    dev_from_4 += np.abs(v1.degree() - 4)
                degree_cost = self.lambda_degree * dev_from_4
                path_cost += degree_cost
                path_cost_components['degree'] = path_cost_components.get('degree', 0) + degree_cost

            if self.lambda_turn_angle > 0:
                if i >= 1:
                    p0 = path_seq[i-1]
                    v0 = g.vs.select(name=p0)[0]
                    # We penalize deviations from 180 (i.e. straight)
                    turn_angle = g.vs[v1.index]['angles'][(v0.index, v2.index)]
                    turn_angle = np.abs(turn_angle - 180)
                    turn_angle_cost = self.lambda_turn_angle * turn_angle
                    path_cost += turn_angle_cost
                    path_cost_components['turn_angle'] = path_cost_components.get('turn_angle', 0) + turn_angle_cost
            
            i += 1

        if return_compnents:
            return path_cost, path_cost_components
        else:
            return path_cost 




        # # PATH LENGTH 2
        # if path_length == 2:
        #     p0, p1 = path_seq
        #     v0 = g.vs.select(name=p0)[0]
        #     v1 = g.vs.select(name=p1)[0]
            
        #     if self.lambda_dist > 0:
        #         # Distance is Euclidean distance
        #         distance = g.es.select(_between=([v0.index], [v1.index]))['eucl_dist'][0]
                
        #         # Operationalize width as way to mitigate distance
        #         if self.lambda_width > 0:
        #             dist_cost = (self.lambda_dist * distance / (self.lambda_width * width))
        #         else:
        #             dist_cost = self.lambda_dist * distance
        #         path_cost_components['dist'] = dist_cost 
        #         path_cost += dist_cost

        #     if self.lambda_dist > 0:
        #         # Operationalize nodal degree as deviation from 4 way intersection
        #         degree_dif_from_4 = np.abs()

        #     if self.lambda_width > 0:
        #         pass 

        # # PATH LENGTH > 2
        # else:
        #     i = 0
        #     for p0, p1, p2 in zip(path_seq[0:-2], path_seq[1:-1], path_seq[2:]):
        #         v0 = g.vs.select(name=p0)[0]
        #         v1 = g.vs.select(name=p1)[0]
        #         v2 = g.vs.select(name=p2)[0]

        #         if self.lambda_dist > 0:
        #             # Distance is Euclidean distance
        #             distance = g.es.select(_between=([v1.index], [v2.index]))['eucl_dist'][0]
        #             if i == 0:
        #                 distance += g.es.select(_between=([v0.index], [v1.index]))['eucl_dist'][0]

        #             # Operationalize width as way to mitigate distance
        #             if self.lambda_width > 0:
        #                 dist_cost = (self.lambda_dist * distance / (self.lambda_width * width))
        #                 path_cost += dist_cost
        #             else:
        #                 dist_cost = self.lambda_dist * distance
        #                 path_cost += dist_cost 
        #             path_cost_components['dist'] = dist_cost 

        #         if self.lambda_turn_angle > 0:
        #             # We penalize deviations from 180 (i.e. straight)
        #             turn_angle = g.vs[v1.index]['angles'][(v0.index, v2.index)]
        #             turn_angle = np.abs(turn_angle - 180)
        #             path_cost += self.lambda_turn_angle * turn_angle
        #             path_cost_components['turn_angle'] = self.lambda_turn_angle * turn_angle 

        #         if self.lambda_degree > 0:
        #             pass

        #         if self.lambda_width > 0:
        #             pass 

        #         i += 1
        
        # if return_compnents:
        #     return path_cost, path_cost_components
        # else:
        #     return path_cost 











