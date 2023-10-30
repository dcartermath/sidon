import time
import itertools as it
import numpy as np
import scipy.optimize
#We use extensive caching to speed up computation.
from functools import cache

def f(a,b,C):
    '''Helper function: f(a,b,C) = min_(a <= z <= b) (z - C)^2.'''
    if a >= C:
        return (a-C)**2
    if b <= C:
        return (b-C)**2
    return 0

def b1(tau, alpha, w):
    '''The bound from Lemma 3.1.'''
    return tau + 1/tau \
           - 2*tau*sum((w[j+1] - w[j])*f(alpha[j], alpha[j+1], 1)
                       for j in range(len(alpha) - 1))

def b2(tau, alpha, c, w):
    '''The bound from Lemma 3.2.'''
    p = []
    for wj in w:
        if 0 <= wj <= 1:
            p.append((wj, 'r'))
        if 0 <= wj+c <= 1:
            p.append((wj+c, 'q'))
    p = sorted(p)
    #Adding a dummy value since zeta and eta are 1-indexed.
    zeta = [None]
    eta = [None]
    j_zeta = 0
    j_eta = 0
    for _, letter in p:
        if letter == 'r':
            j_eta += 1
        if letter == 'q':
            j_zeta += 1
        zeta.append(j_zeta)
        eta.append(j_eta)
    return c*tau + 1/(c*tau) \
           - 2*tau/c**2 \
           *sum((p[j][0] - p[j-1][0])
                *f(alpha[eta[j]-1] - alpha[zeta[j]],
                   alpha[eta[j]] - alpha[max(zeta[j]-1,0)],
                   c)
                for j in range(1, len(p)))
    #eta[j] >= 1 for all j since p[0]=r[0], so we only need
    # to special-case for zeta[j] == 0 above.

def problem(K, bound_ineq, objective=None, additional_constraints=None):
    '''Linear program with bounds given by bound_ineq (see code comments below).
If objective is not specified, tests feasibility of the linear program.'''
    if objective is None:
        objective = [0]*(K+2)
    if additional_constraints is None:
        additional_constraints = []
    #The constraint matrix and vector. We need A_ub @ w <= b_ub.
    A_ub = []
    b_ub = []
    #(x, y, bound) in bound_ineq means w[y] - w[x] <= bound
    for (x, y, bound) in bound_ineq:
        to_add = [0]*(K+2)
        to_add[x] += -1
        to_add[y] += 1
        A_ub.append(to_add)
        b_ub.append(bound)
    
    for add_constraint in additional_constraints:
        A_ub.append(list(add_constraint))
        b_ub.append(0)#add_constraint[-1])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    #additionally, we have w[0] = 0, w[1] = 1.
    A_eq = np.array([[1] + [0]*(K+1),
                     [0]*(K+1) + [1]])
    b_eq = np.array([0, 1])

    return scipy.optimize.linprog(objective, A_ub, b_ub, A_eq, b_eq, (0,1))

@cache
def s_to_p(s):
    '''Returns the order of the q's and r's in P, given s. Also returns
the first index of q that is greater than 1.'''
    p = [('r', j) for j in range(len(s))]
    for j, q_pos in enumerate(s):
        if q_pos >= len(s) - 1:
            end_q = j
            break
        p.insert(p.index(('r', q_pos+1)), ('q', j))
    return p, end_q

@cache
def s_to_boundary(s, c):
    '''Given s, returns the bound_ineq needed to call problem. This encodes
the cell boundary inequalities.'''
    p, end_q = s_to_p(s)
    bound_ineq = []
    #p[j] <= p[j+1]
    for j in range(len(p) - 1):
        x = p[j+1][1]
        y = p[j][1]
        bound = 0
        if p[j][0] == 'q':
            bound -= c
        if p[j+1][0] == 'q':
            bound += c
        bound_ineq.append((x, y, bound))
    #q[end_q] >= p[L]
    bound_ineq.append((end_q, len(s) - 1, c))
    return bound_ineq

def _feasible_cells(c, K):
    '''Given c and K, generates the list of nonempty cells (values of s).
This is a generator.'''
    for s in it.product(*(range(j, K+2) for j in range(K+2))):
        if not all(s[j] <= s[j+1] for j in range(K+1)):
            continue
        bound_ineq = s_to_boundary(s, c)
        prob = problem(K, bound_ineq)
        if prob.status == 0: #feasible
            yield s
        else:
            assert prob.status == 2 #check for errors other than infeasibility

@cache
def feasible_cells(c, K):
    '''Given c and K, generates the list of nonempty cells (values of s).'''
    return list(_feasible_cells(c, K))

@cache
def b1_coeffs(tau, alpha):
    K = len(alpha) - 2
    out = [0]*(K+1) + [tau + 1/tau]
    for j in range(K+1):
        out[j+1] += -2*tau*f(alpha[j], alpha[j+1], 1)
        out[j] -= -2*tau*f(alpha[j], alpha[j+1], 1)
    return out

@cache
def s_to_b2_coeffs(s, tau, alpha, c):
    ''' The bound from Lemma 3.2 in cell s, presented as a list a,
with b_inf <= a^T w.'''
    K = len(alpha) - 2
    p, end_q = s_to_p(s)
    zeta = [None]
    eta = [None]
    j_zeta = 0
    j_eta = 0
    for letter, _ in p:
        if letter == 'r':
            j_eta += 1
        if letter == 'q':
            j_zeta += 1
        zeta.append(j_zeta)
        eta.append(j_eta)
    
    out = [0]*(K+1) + [c*tau + 1/(c*tau)]
    for j in range(1, len(p)):
        factor = -2*tau/c**2\
                 * f(alpha[eta[j]-1] - alpha[zeta[j]],
                     alpha[eta[j]] - alpha[max(zeta[j]-1,0)],
                     c)
        out[p[j][1]] += factor
        out[p[j-1][1]] -= factor
        if p[j][0] == 'q':
            out[-1] += factor * c
        if p[j-1][0] == 'q':
            out[-1] -= factor * c
    return out

def multioptimize(K, boundary, objectives):
    '''Given the list of lists of boundary inequalities (in bound_ineq form)
of a cell intersection and some number of objective functions, return the maximum
over objectives of the minimum over the cell of the objective value. Also returns
the maximizing point.'''
    worst_bound = 0
    at = None
    cases = 0
    #Choose which objective in the maximum
    for wi, winner in enumerate(objectives):
        additional_constraints = []
        for li, loser in enumerate(objectives):
            if li == wi:
                continue
            #winner >= all losers
            additional_constraints.append(tuple(x - y for x,y in zip(winner, loser)))
        prob = problem(K, boundary, tuple(-x for x in winner), tuple(additional_constraints))
        if prob.status == 0: #feasible
            cases += 1
            bound = -prob.fun
            if bound > worst_bound:
                worst_bound = bound
                at = prob.x
        else:
            assert prob.status == 2 #check for errors other than infeasibility
    return worst_bound, at, cases

#Some helper functions for cell intersections
@cache
def cell_intersection_nonempty(K, bound_ineqs):
    '''Determine if the intersection of several cells is nonempty.'''
    prob = problem(K, tuple(x for y in bound_ineqs for x in y))
    if prob.status == 0:
        return True
    else:
        assert prob.status == 2

@cache
def is_valid_cell_pair(c1, c2, K, a, b):
    '''Determine if the intersection of the a'th feasible cell with c=c1 and the
b'th feasible cell with c=c2 is nonempty.'''
    #We are heavily relying on the feasible_cells caching here.
    to_test = (tuple(s_to_boundary(feasible_cells(c1, K)[a], c1)),
               tuple(s_to_boundary(feasible_cells(c2, K)[b], c2)))
    return cell_intersection_nonempty(K, to_test)

@cache
def combined_bound(tau, alpha, cs, printing=False):
    '''Given parameters, return the final bound on b_inf and the worst case w.'''
    K = len(alpha) - 2
    #Determine the nonempty cell intersections.
    # We build these one cell at a time, checking the pairwise intersections
    # for nonemptiness. At the end, we check the full cell intersection, but
    # the pairwise check gets rid of the vast majority of empty intersections.
    base_cells = [feasible_cells(c, K) for c in cs]
    if printing:
        for i in range(len(cs)):
            print(f'c=c{i+1} has {len(base_cells[i])} nonempty cells')
            
    cell_intersections = [()] #The values of s resulting in the intersection.
    #The indices into base_cells giving the nonempty intersections.
    cell_indices = [()]
    for i in range(len(base_cells)):
        new_cell_intersections = []
        new_cell_indices = []
        for cur_a, cur_cell_intersection in zip(cell_indices, cell_intersections):
            for a, new_cell in enumerate(base_cells[i]):
                if not all(is_valid_cell_pair(cs[i], cs[j], K, a, b)
                           for (j,b) in enumerate(cur_a)):
                    continue
                new_cell_intersections.append(cur_cell_intersection + (new_cell,))
                new_cell_indices.append(cur_a + (a,))
        cell_intersections = new_cell_intersections
        cell_indices = new_cell_indices
    final_cell_intersections = []
    for cell_intersection in cell_intersections:
        to_test = tuple(sorted(tuple(s_to_boundary(s, c))
                               for s, c in zip(cell_intersection, cs)))
        #(sort for caching purposes; the order of the bounds doesn't affect the LP)
        if cell_intersection_nonempty(K, to_test):
            final_cell_intersections.append(cell_intersection)
    if printing:
        print(f'There are {len(final_cell_intersections)} nonempty cell intersections')
    
    #Find the worst b_inf bound over the cell intersections
    worst_bound = 0
    at = None
    b1_c = b1_coeffs(tau, alpha)
    total_cases = 0
    for cell_intersection in final_cell_intersections:
        objectives = [b1_c]
        for i,c in enumerate(cs):
            s = cell_intersection[i]
            objectives.append(s_to_b2_coeffs(s, tau, alpha, c))
        bound, x, cases = multioptimize(K,
                                        tuple(bound_ineq
                                              for s,c in zip(cell_intersection, cs)
                                              for bound_ineq in s_to_boundary(s, c)),
                                        tuple(objectives))
        total_cases += cases
        if bound > worst_bound:
            worst_bound = bound
            at = x
    if printing:
        print(f'There are {total_cases} total (feasible) cases')
    return worst_bound, at

start_time = time.time()

# Theorem 2.1
tau = 1.07950
alpha = (0, 0.72720, 1.31609, float('inf'))
cs = (0.86838,)

cb, at = combined_bound(tau, alpha, cs)
print(f'Theorem 2.1 gives b_inf <= {cb}\n')

# Theorem 3.3
print('Theorem 3.3 data:')
tau = 1.12733
alpha = (0, 0.70749, 0.78822, 0.87175, 1.12464, 1.18020, 1.24610, float('inf'))
cs = (0.66461, 0.67780, 0.71884)
cb, at = combined_bound(tau, alpha, cs, True)
w = tuple(at)
print(f'\nTheorem 3.3 gives b_inf <= {cb}')
print(f'Worst case w is {w}')
print(f'Lemma 3.1 at this w: {b1(tau, alpha, w)}')
for j in range(len(cs)):
    print(f'Lemma 3.2 at this w with c=c{j+1}: {b2(tau, alpha, cs[j], w)}')
print(f'Time taken: {time.time() - start_time} seconds')
