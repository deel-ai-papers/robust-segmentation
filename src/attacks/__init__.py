from functools import partial
from .dense_adversary import dag
from .asma import asma
from .primal_dual_gradient_descent import (
    pdgd,
    pdpgd,
)


def get_attack(epsilon, p, num_steps):
    dict_attack_norms = {
        "pdgd": [2],
        "pdpgd": [2, float("inf")],
        "dag": [2, float("inf")],
        "asma": [2],
    }
    attack_dict = {
        "pdgd": partial(pdgd, num_steps=num_steps),
        "pdpgd": partial(pdpgd, norm=p, num_steps=num_steps),
        "dag": partial(dag, p=p, max_iter=num_steps),
        "asma": partial(asma, num_steps=num_steps),
    }
    list_attacks = [a for n, a in attack_dict.items() if p in dict_attack_norms[n]]
    list_threshold = [
        True if n in ["dag", "asma", "pdgd", "pdpgd"] else False
        for n, a in attack_dict.items()
    ]
    return list_attacks, list_threshold
