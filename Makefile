black:
	@black --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do black "$$src"; done

flake8:
	@flake8 --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do flake8 "$$src"; done

clean:
	@jupyter nbconvert --version
	@find . -maxdepth 3 -name '*.ipynb'\
		| while read -r src; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$$src"; done

all:black flake8
