BGROUND :: background subtraction for XY-data
---------------------------------------------

* BGROUND performs background subtraction for XY-data.
	- XY-data = usually a file with two (or more) columns - (X-data,Y-data).
	- The user can define the columns with XY-data, file header, comments etc.
* Fully automated background subtraction:
	- The user defines input data and one of automatic bkgr subtraction methods.
	- BGROUND reads the data and subtracts the background.
* Semi-automated background subtraction:
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

* [Worked example](https://drive.google.com/file/d/15kHdMp8PUv8rna-qFwVENhjYGRXsxmHE/view?usp=sharing)
  shows the BGROUND package in action.
* [Help on GitHub](https://mirekslouf.github.io/bground/docs/)
  with complete
  [package documentation](https://mirekslouf.github.io/bground/docs/pdoc.html/bground.html)
  and
  [additional examples](https://drive.google.com/drive/folders/1ET-rEUn-G8W1QffAnDhHEJdmJaBPNITp?usp=sharing).


Additional resources
--------------------

* [PyPI](https://pypi.org/project/bground) repository -
  the stable version to install.
* [GitHub](https://github.com/mirekslouf/bground) repository - 
  the current version under development.
* [GitHub Pages](https://mirekslouf.github.io/bground/) -
  the more user-friendly version of GitHub website.


Versions of BGROUND
-------------------

* Version 0.0.1 = an incomplete testing version
* Version 0.0.2 = the basic algorithm works
* Version 0.0.3 = a small improvement of code and docstrings
* Version 0.1 = OO-interface, better arrangement of funcs + semi-complete docs
* Version 0.2 = improved OO-implementation + better UI (commands, saving, help)
* Version 1.0 = finalized version 1, fully working, and completely documented
* Version 1.1 = prepared for multiple background subtraction algorithms
* Version 1.2 = 3 bkg methods: InteractivePlot, RestoreFromPoints, SimpleFunc
* Version 1.3 = 4 bkg methods (added: Baselines method with PyBaseLines algs)


Acknowledgement
---------------

The development was co-funded by TACR, program NCK,
project [TN02000020](https://www.isibrno.cz/en/centre-advanced-electron-and-photonic-optics).
