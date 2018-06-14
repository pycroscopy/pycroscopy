Guidelines for Contribution
============================

**This document is under construction**

Structuring code
----------------

We would like to thank you and several others who have offered their code. We are more than happy to add your code to this project. Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others. We do NOT ask that your code be as efficient as possible. Instead, we have some simpler and easier requests. We have compiled a list of best practices below with links to additional information. If you are confused or need more help, please feel free to `contact us <./contact.html>`_:

* Encapsulate independent sections of your code into functions that can be used individually if required.
* Ensure that your code (functions) is well documented (`numpy format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_) - expected inputs and outputs, purpose of functions
* Please avoid naming variables with single alphabets like ``i`` or ``k``. This makes it challenging to find and fix bugs.
* Ensure that your code works in python 2.7 and python 3.5 (ideally using packages that are easy to install on Windows, Mac, and Linux). It is quite likely that Anaconda (upon which pycroscopy is based) has a comprehensive list packages for science and data that handle most needs. We recommend sticking to Anaconda packages if possible.
* Please ensure that your code files fit into our `package structure <./package_structure.html>`_ (``io``, ``analysis``, ``processing``, ``simulation``, ``viz``) + external folder for jupyter notebook.
* Provide a few examples on how one might use your code - preferably via a jupyter notebook.
* If possible try to limit yourself to packages that are within the standard Anaconda distribution. There is a very good chance that you already satisfy this recommendation. If this is not possible, try to use packages that are easy to to install (pip install). If even this is not possible, try to use packages that at least have conda installers.
* Follow best practices for `PEP8 compatibility <https://www.datacamp.com/community/tutorials/pep8-tutorial-python-code>`_. The easiest way to ensure compatibility is to set it up in your development environment. `PyCharm <https://blog.jetbrains.com/pycharm/2013/02/long-awaited-pep-8-checks-on-the-fly-improved-doctest-support-and-more-in-pycharm-2-7/>`_ does this by default. So, as long as PyCharm does not raise many warning, your code is beautiful!

You can look at our code in our `GitHub project <https://github.com/pycroscopy/pycroscopy>`_ to get an idea of how we organize, document, and submit our code.

Contributing code
-----------------
We recommend that you follow the steps below. Again, if you are ever need help, please contact us:

1. Learn ``git`` if you are not already familiar with it. See our `compilation of tutorials and guides <./external_guides.html>`_, especially `this one <https://github.com/pycroscopy/pycroscopy/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_.
2. Create a ``fork`` of pycroscopy - this creates a separate copy of the entire pycroscopy repository under your user ID. For more information see `instructions here <https://help.github.com/articles/fork-a-repo/>`_.
3. Once inside your own fork, you can either work directly off ``master`` or create a new branch.
4. Add / modify code
5. ``Commit`` your changes (equivalent to saving locally on your laptop). Do this regularly.
6. Repeat steps 4-5. After you reach a certain milestone, ``push`` your commits to your ``remote branch``. To avoid losing work due to problems with your computer, consider ``pushing commits`` once at least every day / every few days.
7. Repeat steps 3-5 till you are ready to have your code added to the parent pycroscopy repository. At this point, `create a pull request <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_. Someone on the development team will review your ``pull request`` and then ``merge`` these changes to ``master``.
