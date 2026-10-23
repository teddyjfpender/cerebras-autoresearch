PYTHON ?= python3
RUFF = uvx --from ruff==0.17.1 ruff

.PHONY: check test lint format list smoke sdk-test host-numerics
check: lint test

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYTHON) research.py validate
	$(PYTHON) research.py ledger verify
	$(PYTHON) -m compileall -q src benchmarks/drivers tools research.py
	bash -n tools/sdk-run tools/sdk-unpacked tools/unpacked-sdk-bin/*

test:
	$(PYTHON) -m unittest discover -s tests -t .

format:
	$(RUFF) check . --fix
	$(RUFF) format .

list:
	$(PYTHON) research.py list

smoke:
	tools/sdk-run $(PYTHON) research.py suite smoke --screen --max-seconds 1200

sdk-test:
	tools/sdk-run cs_python tests/integration/check_attention_contracts.py
	tools/sdk-run cs_python tests/integration/check_driver_contracts.py
	tools/sdk-run cs_python tests/integration/check_replay_contracts.py
	tools/sdk-run cs_python tests/integration/check_reusable_gemm_contracts.py
	tools/sdk-run cs_python tests/integration/check_snapshot_contracts.py

host-numerics:
	$(PYTHON) tests/integration/check_attention_contracts.py
