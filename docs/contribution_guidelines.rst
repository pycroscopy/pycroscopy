Guidelines for Contribution
============================

**This document is under construction**

Structuring code
----------------

We would like to thank you and several others who have offered their code. We are more than happy to add your code to this project. Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others. We do NOT ask that your code be as efficient as possible. Instead, we have some simpler and easier requests:

* Encapsulate independent sections of your code into functions that can be used individually if required.
* Ensure that your code (functions) is well documented (`numpy format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_) - expected inputs and outputs, purpose of functions
* Ensure that your code works in python 2.7 and python 3.5 (ideally using packages that are easy to install on Windows, Mac, and Linux). It is quite likely that Anaconda (upon which pycroscopy is based) has a comprehensive list packages for science and data that handle most needs. We recommend sticking to Anaconda packages if possible.
* Provide a few examples on how one might use your code - preferably via a jupyter notebook.

You can look at our code in our `GitHub project <https://github.com/pycroscopy/pycroscopy>`_ to get an idea of how we organize, document, and submit our code.

Contributing code
-----------------
We recommend that you follow these steps:

1. Learn ``git`` if you are not already familiar with it. See our `compilation of tutorials and guides <./external_guides.html>`_, especially `this one <https://github.com/pycroscopy/pycroscopy/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_.
2. Create your own branch off ``master``
3. Add / modify code
4. ``Commit`` your changes (equivalent to saving locally on your laptop). Do this regularly.
5. Repeat steps 3-4. After you reach a certain milestone, ``push`` your commits to your ``remote branch``. To avoid losing work due to problems with your computer, consider ``pushing commits`` once at least every day / every few days.
6. Repeat steps 3-5 till you are ready to have your code added to the ``master`` branch. At this point, do a ``pull request``. Someone on the development team will review your changes and then ``merge`` these changes to ``master``.
