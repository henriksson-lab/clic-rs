bench:
	bash benchmark/run.sh

loc:
	find src -name '*.rs' | xargs wc -l
	find kernels -name '*.cl' | xargs wc -l

