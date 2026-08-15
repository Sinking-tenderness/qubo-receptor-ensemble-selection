# Stage70 constraint-aware QUBO encoding

## Question

Can the frozen Stage68 quality-floor objective be represented with a materially smaller coefficient range without changing the Stage69 scale-511 feasible problem?

## Encoding

For a fixed subset size $k$, Stage70 uses

$$
E(x,s)=R(x)+P_k\left(\sum_i x_i-k\right)^2
+P_q\left[\sum_i(d_i-c)x_i+\sum_jw_js_j-(D-kc)\right]^2.
$$

The slack range is tightened to $S_{\max}=D-\sum_{r=1}^k d_{(r)}$. Both penalties are set to the known feasible pair-off redundancy upper bound plus one. The integer center $c$ is selected only by the expanded QUBO coefficient range; no holdout metric or selected subset enters this choice.

## Result

- Selected encoding: `tight_cap16_centered_pair_upper`.
- Maximum coefficient dynamic range: `180349`.
- Improvement versus Stage69 scale 511: `4.150x`.
- Maximum logical variables: `101`.
- Maximum quadratic coefficients: `5050`.
- Analytic exact-penalty certificates: `80/80`.
- Source scale-511 exact subset matches versus continuous Stage68: `78/80`.

## Decision boundary

- Compact logical QUBO freeze authorized: `True`.
- Coefficient-noise simulation authorized: `True`.
- Direct-QPU precision gate: `False`.
- Direct-QPU execution authorized: `False`.

This is a post-hoc encoding result on four consumed development targets. It does not establish new-target efficacy, hardware sampling quality, speedup, or quantum advantage.
