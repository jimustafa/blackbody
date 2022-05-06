.PHONY: all
all: dev-requirements.txt

.PHONY: install
install: dev-requirements.txt
	pip-sync $^

dev-requirements.txt: dev-requirements.in
	pip-compile $<

.PHONY: clean
clean:
	- rm *.txt
