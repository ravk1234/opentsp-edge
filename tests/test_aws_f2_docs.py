from pathlib import Path

def test_aws_f2_docs_exist():
    assert Path("docs/aws_f2_bringup.md").exists()

def test_aws_f2_directory_exists():
    assert Path("fpga/aws_f2").exists()

def test_runtime_plan_exists():
    assert Path("fpga/aws_f2/host_runtime_plan.md").exists()