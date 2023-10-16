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
        self.spin = indx.spin
        offset = int(indx.spin == 'b')
        self.amps = [{'T1': amp,
                      'occ': 2 * i + offset,
                      'virt': 2 * a + offset}
                      for i in indx.occ for a in indx.virt
                      if np.log10(np.abs(amp := t1[i, a - n_occ])) > -10]

class T2:
    def __init__(self, indx_1, indx_2, t2, n_occ):
        self.spin1 = indx_1.spin
        self.spin2 = indx_2.spin
        offset1 = int(indx_1.spin == 'b')
        offset2 = int(indx_2.spin == 'b')
        self.amps = [{'T2': amp,
                      'occ': [2 * i + offset1, 2 * j + offset2],
                      'virt': [2 * a + offset1, 2 * b + offset2]}
                     for i in indx_1.occ for j in indx_2.occ
                     for a in indx_1.virt for b in indx_2.virt
                     if (any([indx_1.spin != indx_2.spin, j > i and b > a])
                         and np.log10(np.abs(
                             amp := t2[i, j, a - n_occ, b - n_occ])) > -10)]

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
    
    t1_prods = (comb2prod(comb, 'T1') for comb in combinations(t1_amps, n1)
                if filter_combs(comb, 't1') == 2 * n1)
    
    t2_amps = t2_aa.amps + t2_bb.amps + t2_ab.amps

    t2_prods = (comb2prod(comb, 'T2') for comb in combinations(t2_amps, n2)
                if filter_combs(comb, 't2') == 4 * n2)

    total_prod = [prod_1["Product"] * prod_2["Product"]
                  for prod_1 in t1_prods for prod_2 in t2_prods
                  if (filter_orbs([*prod_1['occ'], *prod_1['virt']],
                                  [*prod_2['occ'], *prod_2['virt']]))]

    return sum(total_prod)

def comb2prod(comb, tn):
    orbs_occ = [entry['occ'] for entry in comb]
    orbs_virt = [entry['virt'] for entry in comb]
    if tn == 'T2':
        orbs_occ = sum(orbs_occ, [])
        orbs_virt = sum(orbs_virt, [])
    return {"Product": np.prod([entry[tn] for entry in comb]),
            'occ': orbs_occ,
            'virt': orbs_virt}

def filter_combs(combination, tn):
    if tn == 't1':
        return len({orb for entry in combination
                    for orb in [entry['occ'], entry['virt']]})
    elif tn == 't2':
        return len({orb for entry in combination
                    for orb in [*entry['occ'], *entry['virt']]})

def filter_orbs(ind_1,ind_2):
    return not bool(set(ind_1).intersection(set(ind_2)))

def mapper(func, *args, **kwargs):
    return list(map(lambda var: func(*var, **kwargs), *args))


