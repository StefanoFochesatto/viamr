.PHONY: clean

test:
	@python -B -m pytest .  # -B option effectively reloads from source (instead of using?/writing cache)

clean:
	@rm -rf __pycache__/ .pytest_cache/ *.egg-info/
	@rm -rf tests/__pycache__/ viamr/__pycache__/
	@rm -rf .coverage htmlcov/
