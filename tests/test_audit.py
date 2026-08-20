import json

from peyk import scoring
from peyk.audit import Policy, audit, check, passed
from peyk.cli import main
from peyk.models import Accelerator, HardwareProfile, ModelVariant


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores_physical=16,
                           cpu_cores_logical=32, ram_total_gb=64, ram_available_gb=48,
                           accelerator=Accelerator.NONE, mem_bandwidth_gbs=50)


def _scored(family="Qwen2.5", params=7, license="apache-2.0", langs=("multi",)):
    v = ModelVariant(model_id="x", family=family, params_b=params, file_size_gb=4.0,
                     license=license, languages=list(langs))
    return scoring.score_variant(v, _hw(), 8192, ["en"], scoring.weights_for(None))


def test_check_license_violation():
    s = _scored(license="gemma")
    reasons = check(s, Policy(allow_licenses={"apache-2.0", "mit"}))
    assert any("license" in r for r in reasons)


def test_check_max_params():
    s = _scored(params=70)
    assert any("params" in r for r in check(s, Policy(max_params_b=32)))


def test_check_language_requirement():
    s = _scored(langs=("en",))
    assert any("language" in r for r in check(s, Policy(require_languages=["tr"])))
    multi = _scored(langs=("multi",))
    assert check(multi, Policy(require_languages=["tr"])) == []


def test_compliant_when_all_pass():
    s = _scored(family="Qwen2.5", params=7, license="apache-2.0", langs=("multi",))
    assert check(s, Policy(max_params_b=32, allow_licenses={"apache-2.0"},
                           require_languages=["tr"], min_quality=50)) == []


def test_audit_sorts_compliant_first_and_passed():
    rows = audit([_scored(license="gemma"), _scored(license="apache-2.0")],
                 Policy(allow_licenses={"apache-2.0"}))
    assert rows[0].compliant is True
    assert passed(rows) is True


def test_policy_from_file(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"max_params_b": 14, "allow_licenses": ["MIT"],
                             "require_languages": ["tr"], "min_quality": 60}))
    policy = Policy.from_file(str(p))
    assert policy.max_params_b == 14
    assert policy.allow_licenses == {"mit"}  # lowercased


def test_cli_audit_exit_codes():
    assert main(["audit", "--allow-license", "apache-2.0"]) == 0        # some comply
    assert main(["audit", "--allow-license", "no-such-license"]) == 1   # none comply


def test_cli_audit_json(capsys):
    rc = main(["audit", "--max-params", "8", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert "passed" in out and "models" in out
    assert rc == 0
