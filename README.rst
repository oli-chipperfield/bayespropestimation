==================================
Bayesian estimation of proportions
==================================


.. image:: https://img.shields.io/pypi/v/bayespropestimation.svg
        :target: https://pypi.python.org/pypi/bayespropestimation

.. image:: https://img.shields.io/travis/oli-chipperfield/bayespropestimation.svg
        :target: https://travis-ci.com/oli-chipperfield/bayespropestimation

.. image:: https://readthedocs.org/projects/bayespropestimation/badge/?version=latest
        :target: https://bayespropestimation.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Class method for the Bayesian estimation and comparison of proportions


* Free software: MIT license
* Documentation: https://bayespropestimation.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Bayesian estimation of proportions, run this command in your terminal:

.. code-block:: console

    $ pip install bayespropestimation

This is the preferred method to install Bayesian estimation of proportions, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Bayesian estimation of proportions can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/oli-chipperfield/bayespropestimation

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/oli-chipperfield/bayespropestimation/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/oli-chipperfield/bayespropestimation
.. _tarball: https://github.com/oli-chipperfield/bayespropestimation/tarball/master


=====
Usage
=====

To use Bayesian estimation of proportions in a project::

    import bayespropestimation


=====
test math
=====

Test inline equation :math:`\\textrm{Beta}(y_k + \\alpha, n_k - y_k + \\beta)`

Test math block:

.. math::

    p(\\theta_k | Y_k) \\propto L(Y_k | \\theta_k)p(\\theta_k)
    \\propto \\Big( {n_k \\choose y_k} \\theta_k^{y_k} (1 - \\theta_k)^{(n_k - y_k)} \\Big) \\Big(\\frac{\\theta^{(\\alpha -1)}(1 - \\theta_k)^{(\\beta - 1)}}{\\mathbb{B}(\\alpha, \\beta)} \\Big)

Test math block end