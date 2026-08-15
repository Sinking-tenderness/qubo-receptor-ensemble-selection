# Stage69 QUBO precision compression

## Question

How far can the Stage68 quality-floor QUBO coefficient precision be reduced without changing its feasible baseline, selected subsets, or held-out screening behavior?

## Frozen screen

The conservative integer quality constraint was evaluated at scales

$$
q\in\{31,63,127,255,511,1023,2047,4095\}.
$$

For each receptor deficit $d_i$, Stage69 uses $c_i=\lceil qd_i\rceil$ and accepts only states satisfying $\sum_i c_i x_i\le D$. This rounding direction guarantees that every integer-feasible state also satisfies the original continuous Stage68 quality floor.

## Result

- Compression gate passed: `False`.
- Smallest uniformly feasible scale: `511`.
- Feasible cells: `80/80`.
- Exact subset matches: `78/80`.
- Mean subset Jaccard versus continuous Stage68: `0.992262`.
- Mean absolute BEDROC20 gap: `0.000385`.
- Maximum coefficient dynamic range: `748509`.
- Compression factor versus scale 4095: `64.206x`.

## Hardware boundary

- Direct-QPU precision gate: `False`.
- Direct-QPU execution authorized: `False`.
- Compact hybrid or gate-model prototype authorized: `False`.

Stage69 freezes a smaller logical QUBO only when all 80 historical development cells remain feasible and fidelity gates pass. It does not establish embedding feasibility, hardware sampling quality, solver speedup, or quantum advantage.
