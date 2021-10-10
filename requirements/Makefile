.PHONY: all
all: setup.txt dev-requirements.txt

.PHONY: install
install: setup.txt dev-requirements.txt
	pip-sync $^
	pip install -e ../[examples]

setup.txt: ../setup.py
	pip-compile $< --output-file $@

dev-requirements.txt: dev-requirements.in setup.txt
	pip-compile $<

.PHONY: clean
clean:
	- rm *.txt
