all: 
	pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && evince main.pdf

soutenance:
	pdflatex soutenance.tex && evince soutenance.pdf

clean:
	rm $(shell find . -name '*.aux') $(shell find . -name '*.log') $(shell find . -name '*.out') $(shell find . -name '*.toc') $(shell find . -name '*.bbl') $(shell find . -name '*.blg')
