import re
import numpy as np
import cvxpy as cp
from IPython.display import Markdown,display

def printmd(string):
    display(Markdown(string))
    
def extract_quoted_words(string):
    quoted_words = re.findall(r'"([^"]*)"', string)
    return quoted_words    

def finite_difference_matrix(n, dt, order):
    """
    n: number of points
    dt: time interval
    order: (1=velocity, 2=acceleration, 3=jerk)
    """ 
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
        Get matrices to compute velocities, accelerations, and jerks
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def optimization_based_smoothing_1d(
        traj,
        dt=0.1,
        x_init=None,
        x_final=None,
        vel_limit=None,
        acc_limit=None,
        jerk_limit=None,
        p_norm=2,
        ):
    """
        1-D smoothing based on optimization
    """
    n = len(traj)
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    # Convex optimization
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    # Boundary condition
    A_list,b_list = [],[]
    if x_init is not None:
        A_list.append(np.eye(n,n)[0,:])
        b_list.append(x_init)
    if x_final is not None:
        A_list.append(np.eye(n,n)[-1,:])
        b_list.append(x_final)
    # Velocity, acceleration, and jerk limits
    C_list,d_list = [],[]
    if vel_limit is not None:
        C_list.append(A_vel)
        C_list.append(-A_vel)
        d_list.append(vel_limit*np.ones(n))
        d_list.append(vel_limit*np.ones(n))
    if acc_limit is not None:
        C_list.append(A_acc)
        C_list.append(-A_acc)
        d_list.append(acc_limit*np.ones(n))
        d_list.append(acc_limit*np.ones(n))
    if jerk_limit is not None:
        C_list.append(A_jerk)
        C_list.append(-A_jerk)
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n))
    constraints = []
    if A_list:
        A = np.vstack(A_list)
        b = np.hstack(b_list).squeeze()
        constraints.append(A @ x == b) 
    if C_list:
        C = np.vstack(C_list)
        d = np.hstack(d_list).squeeze()
        constraints.append(C @ x <= d)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    traj_smt = x.value
    return traj_smt