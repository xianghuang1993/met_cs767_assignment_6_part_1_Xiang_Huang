# Edited from
# https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html (original)
# and https://medium.com/edureka/bayesian-networks-2499f133d2ec
'''
Problem: Doors A, B, C hide 1 prize, 2 goats. You ("guest") pick one but don't open it.
Monty opens one of the other two that shows a goat. Should you switch?

The code expresses Bayesian network: Guestdoor   Prizedoor
                                            \     /
                                             v  v
                                          Montydoor
'''
from pomegranate import *  # for Bayesian networks

# Door selected by guest is random, e.g., p(A) = 1/3
guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
# Door containing prize is random
prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})

'''
"monty" expresses conditional probabilities, e.g., [B,B,A,0.5]
means: given that the guest selected B and the prize is at B,
Monty Hall will select A with probability 0.5. Formally,
p(monty=A | guest=B AND prize=B) = 0.5
'''
monty = ConditionalProbabilityTable(
[[ 'A', 'A', 'A', 0.0 ],
[ 'A', 'A', 'B', 1.0 ], # edited
[ 'A', 'A', 'C', 0.0 ], # edited
[ 'A', 'B', 'A', 0.0 ],
[ 'A', 'B', 'B', 0.0 ],
[ 'A', 'B', 'C', 1.0 ],
[ 'A', 'C', 'A', 0.0 ],
[ 'A', 'C', 'B', 1.0 ],
[ 'A', 'C', 'C', 0.0 ],
[ 'B', 'A', 'A', 0.0 ],
[ 'B', 'A', 'B', 0.0 ],
[ 'B', 'A', 'C', 1.0 ],
[ 'B', 'B', 'A', 0.5 ],
[ 'B', 'B', 'B', 0.0 ],
[ 'B', 'B', 'C', 0.5 ],
[ 'B', 'C', 'A', 1.0 ],
[ 'B', 'C', 'B', 0.0 ],
[ 'B', 'C', 'C', 0.0 ],
[ 'C', 'A', 'A', 0.0 ],
[ 'C', 'A', 'B', 1.0 ],
[ 'C', 'A', 'C', 0.0 ],
[ 'C', 'B', 'A', 1.0 ],
[ 'C', 'B', 'B', 0.0 ],
[ 'C', 'B', 'C', 0.0 ],
[ 'C', 'C', 'A', 0.0 ], # edited
[ 'C', 'C', 'B', 1.0 ], # edited
[ 'C', 'C', 'C', 0.0 ]], [guest, prize])

# Name nodes ("State"s in pomegranate, with probability distributions)
guest_node = State(guest, name="guest")
prize_node = State(prize, name="prize")
monty_node = State(monty, name="monty")

# Define the Bayesian network
network = BayesianNetwork("Solving the Monty Hall Problem With Bayesian Networks")
network.add_states(guest_node, prize_node, monty_node)
network.add_edge(guest_node, monty_node)
network.add_edge(prize_node, monty_node)
network.bake()  # compile

print("=====Probabilities given guest picked door A=====\n")
beliefs = network.predict_proba({'guest': 'A'})  # the network, given that guest = 'A'
# zip() aggregates elements (see https://docs.python.org/3.3/library/functions.html#zip)
print("n".join("{}t{}".format(state.name, belief)  # print each node and its prob.
               for state, belief in zip(network.states, beliefs)))

print("\n=====Probabilities given guest picked door A and Monty door B=====\n")
beliefs = network.predict_proba({'guest': 'A', 'monty': 'B'})
print("n".join("{}t{}".format(state.name, str(belief))
                for state, belief in zip( network.states, beliefs )))
