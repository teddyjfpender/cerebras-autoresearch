PYTHON ?= python3
RUFF = uvx --from ruff==0.17.1 ruff

.PHONY: check test lint format list smoke sdk-test docs
check: lint test

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYTHON) research.py validate
	$(PYTHON) tools/catalog_docs.py --check
	$(PYTHON) research.py ledger verify
	$(PYTHON) -m compileall -q src benchmarks/drivers tools research.py
	bash -n tools/sdk-run

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
	tools/sdk-run cs_python tests/integration/check_driver_contracts.py
	tools/sdk-run cs_python tests/integration/check_replay_contracts.py
	tools/sdk-run cs_python tests/integration/check_reusable_gemm_contracts.py
	tools/sdk-run cs_python tests/integration/check_snapshot_contracts.py

	tools/sdk-run cs_python tests/integration/check_resident_swiglu_contracts.py
	tools/sdk-run cs_python tests/integration/check_resident_swiglu_replay.py
	tools/sdk-run cs_python tests/integration/check_snapshot_replay_contracts.py
	tools/sdk-run cs_python tests/integration/check_resident_swiglu_bf16_contracts.py
	tools/sdk-run cs_python tests/integration/check_bf16_chain_replay.py
	tools/sdk-run cs_python tests/integration/check_recurrent_gdn_contracts.py
	tools/sdk-run cs_python tests/integration/check_recurrent_gdn_driver.py
	tools/sdk-run cs_python tests/integration/check_recurrent_gdn_replay.py
	tools/sdk-run cs_python tests/integration/check_recurrent_gdn_snapshots.py
	tools/sdk-run cs_python tests/integration/check_snapshot_grid_contracts.py

	tools/sdk-run cs_python tests/integration/check_native_gdn_contracts.py
	tools/sdk-run cs_python tests/integration/check_native_gdn_layout.py
	tools/sdk-run cs_python tests/integration/check_native_gdn_driver.py
	tools/sdk-run cs_python tests/integration/check_native_gdn_replay.py
	tools/sdk-run cs_python tests/integration/check_native_gdn_snapshots.py

	tools/sdk-run cs_python tests/integration/check_full_gdn_contracts.py
	tools/sdk-run cs_python tests/integration/check_full_gdn_layout.py
	tools/sdk-run cs_python tests/integration/check_full_gdn_driver.py
	tools/sdk-run cs_python tests/integration/check_full_gdn_replay.py
	tools/sdk-run cs_python tests/integration/check_full_gdn_snapshots.py

docs:
	$(PYTHON) tools/catalog_docs.py --write
