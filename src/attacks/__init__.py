from functools import partial
from .dense_adversary import dag
from .asma import asma
from .primal_dual_gradient_descent import (
    pdgd,
    pdpgd,
)
from .alma_prox import alma
from .pgd import pgd


def get_attack(epsilon, p, num_steps):
    dict_attack_norms = {
        "pdgd": [2],
        "pdpgd": [2, float("inf")],
        "dag": [2, float("inf")],
        "asma": [2],
        "pgd": [2, float("inf")],
        "alma": [2],
    }
    attack_dict = {
        "pdgd": partial(pdgd, num_steps=num_steps),
        "pdpgd": partial(pdpgd, norm=p, num_steps=num_steps),
        "dag": partial(dag, p=p, max_iter=num_steps),
        "asma": partial(asma, num_steps=num_steps),
        "pgd": partial(pgd, num_steps=num_steps),
        "alma": partial(alma, num_steps=num_steps, lr_init=epsilon / 4),
    }
    list_attacks = [a for n, a in attack_dict.items() if p in dict_attack_norms[n]]
    list_threshold = [
        True if n in ["dag", "asma", "pdgd", "pdpgd", "alma_prox"] else False
        for n, a in attack_dict.items()
    ]
    return list_attacks, list_threshold

