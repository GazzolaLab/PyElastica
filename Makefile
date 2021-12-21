black:
	@black --version
	@black elastica tests


black_check:
	@black --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do black --check "$$src"; done

flake8:
	@flake8 --version
	@flake8 elastica tests


all:black flake8
ci:black_check flake8
