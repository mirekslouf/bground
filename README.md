BGROUND :: semi-automated background subtraction
------------------------------------------------

* BGROUND performs semi-automated background subtraction for XY-data.
	- XY-data = usually a file with two (or more) columns - (X-data,Y-data).
	- The user can define which columns represent the XY-data to process.
* How does it work?
	- BGROUND reads XY-data and shows them in an interactive plot.
	- The user defines background points (with a mouse + keyboard).
	- BGROUND does the rest (background calculation and subtraction).
	
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
  [worked example](https://www.dropbox.com/scl/fi/l3rripp1pjby50law3jbr/01_bground.nb.pdf?rlkey=x8p2dx4eaea3qepm94y485kk7&st=k27iewfw&dl=0)
  to see how BGROUND works.
* Download
  [complete examples with data](https://www.dropbox.com/scl/fo/08ougjp96dnwr1wqqm7be/AIStLY7i0yb80Yq3xKn1blw?rlkey=806nl015x93qte85feldycsxu&dl=0)
  and try BGROUND yourself.

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
