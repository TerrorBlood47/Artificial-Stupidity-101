import itertools
import copy
import random

class VariableNode:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.neighbors = []  # Connected function nodes
        self.incoming = {}
        self.outgoing = {}

    def send_message(self, fnode):
        message = {}
        for xi in self.domain:
            s = 0
            for n in self.neighbors:
                if n is not fnode:
                    s += self.incoming.get(n, {}).get(xi, 0)
            message[xi] = s
        self.outgoing[fnode] = message

    def best_value(self):
        z = {}
        for xi in self.domain:
            total = sum(self.incoming.get(fnode, {}).get(xi, 0) for fnode in self.neighbors)
            z[xi] = total
        best = max(sorted(z), key=lambda x: z[x])
        return best, z

class FunctionNode:
    def __init__(self, name, variables, utility_table):
        self.name = name
        self.variables = variables    # list of variable node names
        self.utility_table = utility_table  # dict: (tuple of values)->utility
        self.neighbors = []           # variable node objects
        self.incoming = {}
        self.outgoing = {}

        # Precompute: For each variable, each domain value, store sorted assignments by utility (descending)
        self.sorted_assignments = {}  # {var_idx: {xi: [(utility, assignment_tuple)]}}
        domain_lists = {vn:None for vn in variables}
        self._utility_table_assignments = list(self.utility_table.items())
        for idx, varname in enumerate(self.variables):
            # Need variable objects to be attached by graph constructor first, do this in setup_domains later!
            pass

    def setup_domains(self):
        # Once neighbors are attached (by graph constructor), fill sorted_assignments.
        # Should be called ONCE after neighbors assigned!
        self.sorted_assignments = {}
        for idx, v in enumerate(self.neighbors):
            val2assigns = {}
            for xi in v.domain:
                matching = []
                for assign, util in self.utility_table.items():
                    if assign[idx] == xi:
                        matching.append( (util, assign) )
                matching.sort(reverse=True)  # sort by descending utility
                val2assigns[xi] = matching
            self.sorted_assignments[idx] = val2assigns

    def gdp_prune(self, var_idx, varnode):
        pruned_ranges = {}
        for xi in varnode.domain:
            assignments = self.sorted_assignments[var_idx][xi]
            if not assignments:
                continue
            Vi = [t[0] for t in assignments]    # utilities, descending
            p = Vi[0]
            # m: sum max from all incoming (except receiving variable)
            m = 0
            for k, v in enumerate(self.neighbors):
                if k == var_idx:
                    continue
                qmsg = v.outgoing[self]
                m += max(qmsg.values())
            # b: sum of actual incoming for max utility assignment (p)
            b = 0
            for k, v in enumerate(self.neighbors):
                if k == var_idx:
                    continue
                qmsg = v.outgoing[self]
                b += qmsg[assignments[0][1][k]]
            t = m - b
            target = p - t
            # Find in sorted Vi the largest index i so that Vi[i] <= target
            qidx = None
            for i, val in enumerate(Vi):
                if val <= target:
                    qidx = i
                    break
            if qidx is None:  # All Vi > target
                qidx = len(Vi) - 1
            q = Vi[qidx]
            # Indices [q, p] (i.e., those for which q <= val <= p)
            idxs = [i for i, val in enumerate(Vi) if q <= val <= p]
            # We store the *assignment* indices into the permutation over domains
            pruned_ranges[xi] = [assignments[i][1] for i in idxs]
        return pruned_ranges

    def send_message(self, varnode):
        idx = self.variables.index(varnode.name)
        pruned_ranges = self.gdp_prune(idx, varnode)
        message = {}
        for xi in varnode.domain:
            best = float("-inf")
            # Only maximize over GDP-pruned assignments for xi
            relevant_assigns = pruned_ranges.get(xi, [])
            for assign in relevant_assigns:
                u = self.utility_table[assign]
                s = 0
                for i, v in enumerate(self.neighbors):
                    if i == idx:
                        continue
                    qmsg = v.outgoing[self]
                    s += qmsg[assign[i]]
                total = u + s
                if total > best:
                    best = total
            message[xi] = best
        self.outgoing[varnode] = message

class MaxSumSolver:
    def __init__(self, variable_nodes, function_nodes):
        self.variables = {v.name: v for v in variable_nodes}
        self.functions = {f.name: f for f in function_nodes}
        for f in function_nodes:
            for v in f.variables:
                f.neighbors.append(self.variables[v])
                self.variables[v].neighbors.append(f)
        # Tell each function node to precompute GDP views now that neighbors are set
        for f in function_nodes:
            f.setup_domains()

    def run(self, iterations=10):
        # Initialize messages to zero
        for v in self.variables.values():
            for f in v.neighbors:
                v.incoming[f] = {x:0 for x in v.domain}
                v.outgoing[f] = {x:0 for x in v.domain}
        for f in self.functions.values():
            for v in f.neighbors:
                f.incoming[v] = {x:0 for x in v.domain}
                f.outgoing[v] = {x:0 for x in v.domain}

        for it in range(iterations):
            for v in self.variables.values():
                for f in v.neighbors:
                    v.send_message(f)
                    f.incoming[v] = copy.deepcopy(v.outgoing[f])
            for f in self.functions.values():
                for v in f.neighbors:
                    f.send_message(v)
                    v.incoming[f] = copy.deepcopy(f.outgoing[v])

        # Decision
        solution = {}
        scores = {}
        for v in self.variables.values():
            best, z = v.best_value()
            solution[v.name] = best
            scores[v.name] = z
        return solution, scores

def make_util_table(var_order, rows):
    return {tuple(row[:-1]): row[-1] for row in rows}

def create_example_factor_graph():
    domains = {'x1':['R','B'], 'x2':['R','B'], 'x3':['R','B'], 'x4':['R','B']}
    variable_nodes = [VariableNode(name, dom) for name, dom in domains.items()]

    U1_table = [['R','R',1], ['R','B',3], ['B','R',2], ['B','B',1]]
    U2_table = [['R','R',1], ['R','B',5], ['B','R',4], ['B','B',2]]
    U3_table = [['R','R',3], ['R','B',4], ['B','R',10], ['B','B',1]]

    fnodes = [
        FunctionNode('U1', ['x1','x3'], make_util_table(['x1','x3'], U1_table)),
        FunctionNode('U2', ['x2','x3'], make_util_table(['x2','x3'], U2_table)),
        FunctionNode('U3', ['x3','x4'], make_util_table(['x3','x4'], U3_table))
    ]

    return variable_nodes, fnodes

def random_factor_graph(n_vars=4, n_funcs=3, domain=['R','B'], seed=None):
    if seed is not None:
        random.seed(seed)
    var_names = [f'x{i+1}' for i in range(n_vars)]
    variables = [VariableNode(name, domain) for name in var_names]

    func_nodes = []
    for j in range(n_funcs):
        factor_size = random.choice([2,2,3])
        vnames = random.sample(var_names, factor_size)
        var_domains = [domain] * factor_size
        table = []
        for assign in itertools.product(*var_domains):
            u = random.randint(1,10)
            table.append(list(assign)+[u])
        func = FunctionNode(f'U{j+1}', vnames, make_util_table(vnames, table))
        func_nodes.append(func)

    return variables, func_nodes

# Example on slides' graph:
variables, fnodes = create_example_factor_graph()
solver = MaxSumSolver(variables, fnodes)
solution, scores = solver.run(iterations=10)
print("Slide Example Solution:")
print(solution)
print(scores)

# # Random graph example:
# variables, fnodes = random_factor_graph(n_vars=6, n_funcs=5, seed=42)
# solver = MaxSumSolver(variables, fnodes)
# solution, scores = solver.run()
# print("Random Example Solution:")
# print(solution)
# print(scores)