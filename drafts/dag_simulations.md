

```
import networkx as nx

var_names = ['A', 'B', 'C', 'D']
parents = [['B', 'C'], [], ['D'], []]

G = nx.DiGraph()
G.add_nodes_from(var_names)
for i, child in enumerate(var_names):
  for parent in parents[i]:
    G.add_edge(parent, child)
    
topsorted_vars = list(nx.algorithms.dag.topological_sort(G))
parents_of_a = list(G.predecessors('A'))
```
