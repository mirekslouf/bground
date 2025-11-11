BGROUND :: background subtraction for XY-data
---------------------------------------------

* BGROUND performs background subtraction for XY-data.
	- XY-data = usually a file with two (or more) columns - (X-data,Y-data).
	- The user can define which columns represent the XY-data to process.
* Semi-automated background subtraction:
	- BGROUND reads XY-data and shows them in an interactive plot.
	- The user defines background points (with a mouse + keyboard).
	- BGROUND does the rest (background calculation and subtraction).
* Fully automated background subtraction:
	- The user defines input data and one of automatic bkgr subtraction methods.
	- BGROUND reads the data and subtracts the background.

Principle
---------
<img src="https://mirekslouf.github.io/bground/docs/assets/bground_principle.png" alt="BGROUND principle" width="600"/>

Installation
------------
* Requirement: Python with sci-modules: numpy, matplotlib, scipy, pandas
* `pip install bground` = standard installation, no other packages needed

Quick start
-----------
* Look at the
  [worked example](https://drive.google.com/file/d/15kHdMp8PUv8rna-qFwVENhjYGRXsxmHE/view?usp=sharing)
  to see how BGROUND works.

Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/bground) repository -
  the stable version to install.
* [GitHub](https://github.com/mirekslouf/bground) repository - 
  the current version under development.
* [GitHub Pages](https://mirekslouf.github.io/bground/)
  with [help](https://mirekslouf.github.io/bground/docs)
  and [complete package documentation](https://mirekslouf.github.io/bground/docs/pdoc.html/bground.html).

Versions of BGROUND
-------------------

* Version 0.0.1 = an incomplete testing version
* Version 0.0.2 = the basic algorithm works
* Version 0.0.3 = a small improvement of code and docstrings
* Version 0.1 = OO-interface, better arrangement of funcs + semi-complete docs
* Version 0.2 = improved OO-implementation + better UI (commands, saving, help)
* Version 1.0 = finalized version 1, fully working, and completely documented
* Version 1.1 = prepared for multiple background subtraction algorithms

Acknowledgement
---------------

The development was co-funded by TACR, program NCK,
project [TN02000020](https://www.isibrno.cz/en/centre-advanced-electron-and-photonic-optics).
