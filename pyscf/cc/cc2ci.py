import numpy as np
from itertools import combinations
from pyscf.fci.addons import cistring

class IndicesX:
    def __init__(self, spin, str_orb, str_gs, n_occ):
        self.spin = spin
        str_x = bin(str_orb ^ str_gs)
        indx_x = np.nonzero(np.array(list(str_x[:1:-1]), dtype=int))[0]
        self.occ = indx_x[indx_x < n_occ]
        self.virt = indx_x[indx_x >= n_occ]

class T1:
    def __init__(self, indx, t1, n_occ):
        self.amps = [{'T1': t1[i, a - n_occ],
                      'occ': 2 * i + (indx.spin == 'b'),
                      'virt': 2 * a + (indx.spin == 'b')}
                      for i in indx.occ for a in indx.virt]
        self.spin = indx.spin
    
class T2:
    def __init__(self, indx_1, indx_2, t2, n_occ):
        self.spin1 = indx_1.spin
        self.spin2 = indx_2.spin
        self.amps = [{'T2': t2[i, j, a - n_occ, b - n_occ],
                      'occ': [2 * i + (indx_1.spin == 'b'), 2 * j + (indx_2.spin == 'b')],
                      'virt': [2 * a + (indx_1.spin == 'b'), 2 * b + (indx_2.spin == 'b')]}
                     for i in indx_1.occ for j in indx_2.occ
                     for a in indx_1.virt for b in indx_2.virt
                     if any([indx_1.spin != indx_2.spin, j > i and b > a])]

def fci_coefs(ccsolver):
    spat_orbs = cistring.gen_strings4orblist(range(ccsolver.nmo),
                                             ccsolver.nocc)
    phi_ccsd = [amp2coef(spat_orba, spat_orbb, spat_orbs[0], ccsolver)
                for spat_orba in spat_orbs for spat_orbb in spat_orbs]
    return phi_ccsd / np.linalg.norm(phi_ccsd)


def amp2coef(str_orba, str_orbb, str_gs, mol):
    indx_a, indx_b = mapper(IndicesX, zip(['a', 'b'], [str_orba, str_orbb]),
                            str_gs=str_gs, n_occ=mol.nocc)
    num_x = len(indx_a.occ) + len(indx_b.occ)

    t1a, t1b = mapper(T1, zip([indx_a, indx_b]), t1=mol.t1, n_occ=mol.nocc)

    t2_idxz = zip([indx_a, indx_b, indx_a],[indx_a, indx_b, indx_b])
    t2aa, t2bb, t2ab = mapper(T2, t2_idxz, t2=mol.t2, n_occ=mol.nocc)

    coeffs = np.sum([combine_amps(t1a, t1b, t2aa, t2bb, t2ab, num_x, k)
                     for k in range(num_x // 2 + 1)])
    return coeffs if num_x > 0 else 1.0

def combine_amps(t1_a, t1_b, t2_aa, t2_bb, t2_ab, num_x, n2):
    n1 = num_x - 2 * n2

    t1_amps = t1_a.amps + t1_b.amps
    t1_combs = (list(comb) for comb in combinations(t1_amps, n1)
                if filter_combs(comb, 'occ', 't1') == filter_combs(comb, 'virt', 't1') == n1)
    t1_prods = [{"Product": np.prod([entry['T1'] for entry in comb]),
                  'occ': [entry['occ'] for entry in comb],
                  'virt': [entry['virt'] for entry in comb]}
                  for comb in t1_combs]
    
    t2_amps = t2_aa.amps + t2_bb.amps + t2_ab.amps
    t2_combs = (list(comb) for comb in list(combinations(t2_amps, n2))
                if filter_combs(comb, 'occ', 't2') == filter_combs(comb, 'virt', 't2') == 2 * n2)
    t2_prods = [{"Product": np.prod([entry['T2'] for entry in comb]),
                  'occ': sum([entry['occ'] for entry in comb], []),
                  'virt': sum([entry['virt'] for entry in comb], [])}
                  for comb in t2_combs]

    total_prod = [prod_1["Product"] * prod_2["Product"]
                  for prod_1 in t1_prods for prod_2 in t2_prods
                  if (filter_orbs(prod_1['occ'], prod_2['occ']) and
                      filter_orbs(prod_1['virt'], prod_2['virt']))]

    return sum(total_prod)


def filter_combs(combination, field, tn):
    vec = [entry[field] for entry in combination]
    if tn == 't2':
        vec = sum(vec, [])
    return len(set(vec))

def filter_orbs(ind_1,ind_2):
    return not bool(set(ind_1).intersection(set(ind_2)))

def ind2array(ind, tn):
    return np.asarray([tn[tuple(elems)] for elems in ind])

def mapper(func, *args, **kwargs):
    return list(map(lambda var: func(*var, **kwargs), *args))


