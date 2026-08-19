all: box.pdf

box.pdf: box.tex box.aux box.bbl
	pdflatex box
	pdflatex box

box.aux: box.tex
	pdflatex box
	bibtex box

box.bbl: box.aux box.bib
	bibtex box

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.lo? *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi *.fls *.fdb_latexmk

