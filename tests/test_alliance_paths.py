from cc_llm_gateway.alliance_paths import (
    hf_hub_weights_parent_name,
    hf_shared_weights_dir,
    project_user_dir,
)


def test_project_user_dir():
    assert project_user_dir("rrg-tayebi", "msina") == "/project/rrg-tayebi/msina"


def test_hf_shared_weights_dir_default_parent():
    cfg = {"slurm_account": "rrg-tayebi", "hf_hub_weights_parent": "cc-llm-hf-weights"}
    assert (
        hf_shared_weights_dir(cfg, "msina", "openai/gpt-oss-120b")
        == "/project/rrg-tayebi/msina/cc-llm-hf-weights/openai--gpt-oss-120b"
    )


def test_hf_shared_weights_dir_uses_config_parent():
    cfg = {"slurm_account": "rrg-tayebi", "hf_hub_weights_parent": "models"}
    assert (
        hf_shared_weights_dir(cfg, "msina", "meta-llama/Llama-3.2-1B-Instruct")
        == "/project/rrg-tayebi/msina/models/meta-llama--Llama-3.2-1B-Instruct"
    )


def test_hf_shared_weights_dir_missing_parent_key():
    cfg = {"slurm_account": "rrg-tayebi"}
    assert hf_hub_weights_parent_name(cfg) == "cc-llm-hf-weights"
    assert (
        hf_shared_weights_dir(cfg, "msina", "org/model")
        == "/project/rrg-tayebi/msina/cc-llm-hf-weights/org--model"
    )


def test_hf_shared_weights_dir_disabled():
    cfg = {"slurm_account": "rrg-tayebi", "hf_hub_weights_parent": ""}
    assert hf_hub_weights_parent_name(cfg) is None
    assert hf_shared_weights_dir(cfg, "msina", "org/model") is None


def test_hf_shared_weights_dir_absolute_model():
    cfg = {"slurm_account": "rrg-tayebi", "hf_hub_weights_parent": "cc-llm-hf-weights"}
    assert hf_shared_weights_dir(cfg, "msina", "/project/rrg-tayebi/msina/foo") is None


def test_hf_shared_weights_dir_requires_account_and_user():
    cfg = {"slurm_account": "rrg-tayebi", "hf_hub_weights_parent": "cc-llm-hf-weights"}
    assert hf_shared_weights_dir(cfg, "", "org/model") is None
    assert hf_shared_weights_dir({}, "msina", "org/model") is None
