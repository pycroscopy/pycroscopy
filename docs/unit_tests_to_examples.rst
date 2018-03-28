=============================================
Converting Unit Tests to Examples
=============================================

Introduction
------------
* While we have some, we do not have enough ``cookbooks`` with examples on how to use functions that will help people become familiar / comfortable with pycroscopy. Here are cookbooks we already have:

  * `plotting utilities <https://pycroscopy.github.io/pycroscopy/auto_examples/user_tutorials/plot_tutorial_01_interacting_w_h5_files.html#sphx-glr-auto-examples-user-tutorials-plot-tutorial-01-interacting-w-h5-files-py>`_
  * `utilities for reading HDF files <https://pycroscopy.github.io/pycroscopy/auto_examples/user_tutorials/plot_tutorial_01_interacting_w_h5_files.html#sphx-glr-auto-examples-user-tutorials-plot-tutorial-01-interacting-w-h5-files-py>`_

* It takes time to write examples and even more time to write ones that work
* ``Unit Tests`` show how a function should be used and how it behaves using examples. Does this sound familiar (above)? 
* In a unit test, we assert that a function behaves in a certain way. In an example, we simply need to show how the function behaves either by printing the output or by plotting some curves, etc. Thus, it is indeed rather straightforward to convert a unit test to an example. 
* Fortunately, we already have unit tests for the core functions of the "new" pycroscopy
  
  * Typically, one may write several unit tests for a single function to cover as many use cases as possible. More coverage = more robust code.
  * We do not need to translate every single unit test to an example, just the ones that best illustrate a function.
* This guide will show you how to convert existing unit tests to examples for cookbooks 


Case 1
------

As a simple example, we will consider tests for ``recommend_cpu_cores()`` present in ``pycroscopy.core.io.io_utils.py``. As the name suggests, the function recommends the number of CPU cores to use for a specific parallel computation (for example fitting N spectra to a function). 

The unit test
~~~~~~~~~~~~~
While it makes sense in theory to use all available CPU cores to solve an embarrassingly parallel problem quickly, there is a time overhead associated with starting up each CPU core and this time loss can outweigh the speedup that can be gained when using multiple cores. Below is an unit test for this very specific scenario. Essentially, we want the function to recommend using a single core when we tell the function that the operation needs to be performed only a few times. You can find this function in lines 30-36 of ``pycroscopy/tests/core/io/test_io_utils.py``

.. code-block:: python
  
  def test_reccomend_cores_few_small_jobs(self):
      num_jobs = 13
      ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
      self.assertEqual(ret_val, 1)
      ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
      self.assertEqual(ret_val, 1)

Since this function accepts a few optional keyword arguments, one can simulate multiple ways in wich this function may be called for this scenario. Here, we will break down what happens in each line of a unit test.

We first set the number of times the operation needs to be performed (``num_jobs``) to something relatively small ~ 13.

.. code-block:: python

  num_jobs = 13

In the first test we only tell the function the number of times the operation needs to be performed (``num_jobs``) and that each job is relatively short (``lengthy_computation=False``).

.. code-block:: python
  
  ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)

We expect the function to behave a certain way. In this case we expect the function to return an integer value that is equal to 1. So, we will then test to make sure (or in programming-speak we ``assert``) that the returned value from the function (``ret_val``) is indeed equal to what we expect (``1``):

.. code-block:: python

  self.assertEqual(ret_val, 1)

The second test is a slight variation of the aforementioned test where we assume that the user also requests the use of all available CPU cores (``requested_cores=MAX_CPU_CORES``). In this case as well, we expect the function to ignore the user's request and recommend the usage of a single core:

.. code-block:: python
  
  ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
  self.assertEqual(ret_val, 1)

Since the two tests are actually very similar to each other, we clubbed both tests under one unit test function. 

The example
~~~~~~~~~~~
Recall that the translation of a unit test to an example only requires that the behavior of the function be shown via a print statement or a plot. When translating this specific unit test, all we need to do is print the returned value from the function instead of asserting that it would be equal to a certain value. In every other way, it is literally copy pasting code. This specific unit test has already been translated to an example `in this example python script <https://github.com/pycroscopy/pycroscopy/blob/unity_dev/examples/dev_tutorials/plot_io_utils.py>`_ but here the excerpt specific to this unit test:

.. code-block:: python

  # Case 3. Far fewer independent and fast computations, and the function is asked if 3 cores is OK. In this case, configuring
  # multiple cores for parallel computations will probably be slower than serial computation with a single core. Hence,
  # the function will recommend the use of only one core in this case.
  requested_cores = 3
  num_jobs = 13
  recommeded_cores = px.io_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=False)
  print('Recommended number of CPU cores for {} independent, FAST, and parallel '
        'computations using the requested {} CPU cores is {}\n'.format(num_jobs, requested_cores, recommeded_cores))
