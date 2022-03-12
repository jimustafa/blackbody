.PHONY: all
all: dev-requirements.txt

.PHONY: install
install: dev-requirements.txt
	pip-sync $^
	pip install -e ../[examples]

dev-requirements.txt: dev-requirements.in
	pip-compile $<

.PHONY: clean
clean:
	- rm *.txt
