===============
Converting Unit Tests to Examples
===============

Unit Tests
----------
These are small examples that can be used to test the behavior of a single function in an isolated manner. Typically, one may write several unit tests for a single function to cover as many use cases as possible. 

As a simple example, we will consider tests for ``recommend_cpu_cores()`` present in ``pycroscopy.core.io.io_utils.py``. As the name suggests, the function recommends the number of CPU cores to use for a specific parallel computation. Below is an unit test for the function where we consider the case when a specific operation (for example fitting N spectra to a function) needs to be performed only a few times. Since this function accepts a few optional keyword arguments, one can simulate multiple ways in wich this function may be called for this scenario. The unit test below tests the function for the aforementioned scenario in three different ways. You can find this function in lines 30-36 of pycroscopy/tests/core/io/test_io_utils.py

.. code-block:: python
  
  def test_reccomend_cores_few_small_jobs(self):
      num_jobs = 13
      ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
      self.assertEqual(ret_val, 1)
      ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
      self.assertEqual(ret_val, 1)

Essentially, 
