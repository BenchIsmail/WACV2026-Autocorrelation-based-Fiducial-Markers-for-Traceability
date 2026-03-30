import autograd.numpy as np
from pymanopt import Problem
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers.trust_regions import TrustRegions
from pymanopt.function import autograd

def homogeneous(y):
    """ Maps \\mathbb{R}^2 to homogeneous coordinates
        input: y \\in \\mathbb{R}^2
        output: (y[0], y[1], 1)
    """
    return np.array([[1,0],[0,1],[0,0]])@y + np.array([0,0,1])

def jacobian(H, yi):
    """ Given a homography H and a point y_i in the deformed image,
        return the Jacobian of the homography at the point that maps to y_i

         input: H \\in \\mathbb{R}^{3\times 3} representing a homography;
               yi \\in \\mathbb{R}^2
        output: \\nabla_{H^{-1} yi} H
    """
    num = H[:2, :2] - np.outer(yi, H[2, :2]) # numerator
    inv_homogeneous = np.linalg.inv(H) @ homogeneous(yi)
    inv_norm = inv_homogeneous/inv_homogeneous[-1]
    den = H[2, :] @ inv_norm # denominator
    return num / den

def embed(H):
    ''' Adds a third column of (0,0,1) to a 3x2 matrix in a differentiable way
        input: a 3x2 matrix
        output: a 3x3 matrix hstack(H, e_3)
    '''
    add = np.zeros((3,3))
    add[2,2] = 1
    return  H@np.array([[1,0,0],[0,1,0]]) + add

def get_cost(As, ys, manifold, ord="fro-2-1"):
    '''returns cost function for optimization
       input: As: sequence of Jacobian observations;
              ys: corresponding patch centers in deformed image;
              manifold: pymanopt manifold object for optimization
       output: sum of squared frobienius norms ('fro') or sum of frobenius norms ('fro-2-1')
               of observation error
    '''
    n = len(As)
    assert n == len(ys)
    @autograd(manifold)
    def cost(H):
        jacobians = np.array([jacobian(embed(H), yi) for yi in ys])
        if ord == 'fro':
            return np.sum([np.sum((As[i]-jacobians[i])**2) for i in range(n)])/n
        if ord == 'fro-2-1':
            return np.sum([np.sqrt(np.sum((As[i]-jacobians[i])**2)) for i in range(n)])/n
    return cost

def optimize(As, ys, ord='fro-2-1', maxouter=150, maxinner=20):

    # get cost function
    manifold = Euclidean(3,2)
    cost = get_cost(As, ys, manifold, ord=ord)

    # run optimization problem in R^6
    problem = Problem(manifold, cost=cost)
    solver = TrustRegions(max_iterations=maxouter)
    X_opt = solver.run(problem, maxinner=maxinner)

    return X_opt.point, X_opt.cost


""" if __name__ == '__main__':
    # Example homography: H = [[1,0,3],[0,2,5],[.2,.4,1]]
    H = np.array([[1,0,4],[0,2,1],[0.0004,-0.0003,1]])
    # Observation positions in deformed image
    ys = [np.array([0,0]), np.array([1,1]), np.array([2,1]), np.array([5,5]), np.array([4,-5]), np.array([5,-5]), np.array([-2,5]), np.array([5,2])]
    # Observations: real jacobian with noise
    As = [jacobian(H, yi)  for yi in ys]
    print(optimize(As, ys)[0],H) """

