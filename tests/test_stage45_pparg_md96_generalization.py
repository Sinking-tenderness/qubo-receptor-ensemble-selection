import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sampled_subsets_are_deterministic_and_unique():
    module = load_script("stage45_diagnose", "scripts/diagnose_stage45_pparg_md96_generalization.py")
    config = {"landscape_sampling": {"k3_sample_count": 20, "seed": 7}}
    first = module.sampled_subsets(8, config)
    second = module.sampled_subsets(8, config)
    assert first == second
    assert len(first[1]) == 8
    assert len(first[2]) == 28
    assert len(first[3]) == 20
    assert len(set(first[3])) == 20


def test_jaccard():
    module = load_script("stage45_diagnose_jaccard", "scripts/diagnose_stage45_pparg_md96_generalization.py")
    assert module.jaccard((1, 2, 3), (2, 3, 4)) == 0.5


def test_independent_audit_passes(tmp_path):
    module = load_script("stage45_audit", "scripts/audit_stage45_pparg_md96_generalization.py")
    result = module.audit(ROOT, tmp_path / "audit.json")
    assert result["status"] == "stage45_pparg_md96_generalization_diagnosis_independent_audit_ok"
    assert result["row_counts"]["coefficient_stability"] == 96
    assert result["k6_gate_recomputed"] is True
