from scripts.solve_discriminative_coverage_qubo import (
    build_discriminative_qubo,
    coefficient_energy,
)


def test_discriminative_qubo_has_expected_energy_terms() -> None:
    rows = [
        {"ligand_id": "a1", "label": "active", "r1": "-10", "r2": "-8"},
        {"ligand_id": "a2", "label": "active", "r1": "-9", "r2": "-7"},
        {"ligand_id": "d1", "label": "decoy", "r1": "-9", "r2": "-6"},
        {"ligand_id": "d2", "label": "decoy", "r1": "-5", "r2": "-7"},
    ]
    qubo = build_discriminative_qubo(rows, ["r1", "r2"], 2)
    assert coefficient_energy(("r1", "r2"), qubo) == sum(
        [
            float(qubo["constant"]),
            float(qubo["linear_coefficients"]["r1"]),
            float(qubo["linear_coefficients"]["r2"]),
            float(qubo["quadratic_coefficients"]["r1__r2"]),
        ]
    )
