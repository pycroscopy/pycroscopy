=============================================
Converting Unit Tests to Examples
=============================================
**Suhas Somnath**

.. contents::

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

Notes
-----
* The unit tests may not have comments for every set of lines as presented here however it is a good idea to add a few helpful comments wherever possible.
* If the purpose of the function is not immediately clear, please consider reading the ``doctoring`` of the function in the source code.
  
  * We have tried to be as meticulous as possible to explain the purpose of each function and each of the parameters in the docstring
  * If you are using Jupyter to test and write examples, you can press the ``Tab`` key four times to get a separate window explaining the function and its inputs
  * If the docstring and the unit tests are still confusing contact Suhas / Chris.

Case 1 - Simple unit test
-------------------------

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

Case 2 - Failure test
---------------------
A major portion of writing unit tests involves writing tests to assert how the function should handle invalid inputs such as providing a string where an integer was expected, etc. It is not necessary to translate such unit tests into examples but it is important to identify and differentiate "success" unit tests (that are worth translating to examples), like the one above, from "failure" test cases, like the one below. In the example below, we intend to test the function: ``px.hdf_utils.get_unit_values()``. 

.. code-block:: python

    def test_get_unit_values_illegal_key(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Does not exist'])

Knowing the inner workings of the specific function are not relevant for this discussion. In the above example, ``'Does not exist'`` is clearly meant to signify an invalid input. The ``self.assertRaises(KeyError):`` only states that we assert that passing such invalid inputs causes the function to throw a ``KeyError``. Should you encounter such a "failure" test case, you may want to ignore it.

Case 3 - Realistic example
--------------------------
Most unit tests will not look as simple as that in Case 1. However, the unit test is likely to have the same components - 

* some set up code to call the function (case 1 did not have much of this) 
* calling the function
* asserting different things about the returned values / created file / plot etc.

The following unit test tests the `pycroscopy.core.io.hdf_utils.link_as_main()` function which aims to link a dataset with four ancillary datasets to make it a `Main` dataset. You will see that the example is actually fairly similar to the unit test despite its complexity.

.. code-block:: python

    def test_link_as_main(self):
        file_path = 'link_as_main.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'],
                         'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
            dset_source_pos_inds = VirtualDataset('PosIndices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
            dset_source_pos_vals = VirtualDataset('PosValues', source_pos_data, dtype=np.float16, attrs=pos_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = VirtualDataset('source_main', source_main_data,
                                              attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''],
                                 'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
            dset_source_spec_inds = VirtualDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                                   attrs=source_spec_attrs)
            dset_source_spec_vals = VirtualDataset('SpecValues', source_spec_data, dtype=np.float16,
                                                   attrs=source_spec_attrs)

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, dset_source_main)
            h5_pos_inds = writer._create_dataset(h5_f, dset_source_pos_inds)
            h5_pos_vals = writer._create_dataset(h5_f, dset_source_pos_vals)
            h5_spec_inds = writer._create_dataset(h5_f, dset_source_spec_inds)
            h5_spec_vals = writer._create_dataset(h5_f, dset_source_spec_vals)

            self.assertFalse(hdf_utils.check_if_main(h5_main))

            # Now need to link as main!
            hdf_utils.link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

            # Finally:
            self.assertTrue(hdf_utils.check_if_main(h5_main))

        os.remove(file_path)

Though it is not absolutely necessary to understand the intricacies of the entire unit test in order to translate this to an example, here is a breakdown of what is happening in the function:

This unit test requires the creation of a HDF5 file. So, the first thing we do is delete the file if it already exists to avoid conflicts. 

.. code-block:: python

  file_path = 'link_as_main.h5'
  self.__delete_existing_file(file_path)

Next, we create the file and open it:

.. code-block:: python

  with h5py.File(file_path) as h5_f:

In this case, the main dataset is a 4D dataset (`X`, `Y` dimensions in positions and `Bias`, `Cycle` spectroscopic dimensions). 

.. code-block:: python

  num_rows = 3
  num_cols = 5
  num_cycles = 2
  num_cycle_pts = 7

First we create the `Position` `Indices` and `Values` datasets

.. code-block:: python

  source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                               np.repeat(np.arange(num_rows), num_cols))).T
  pos_attrs = {'units': ['nm', 'um'],
               'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
  dset_source_pos_inds = VirtualDataset('PosIndices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
  dset_source_pos_vals = VirtualDataset('PosValues', source_pos_data, dtype=np.float16, attrs=pos_attrs)

Next, we prepare the (random) data that will be contained in the Main dataset. To ensure that advanced features such as `region references` are retained, we add two simple region references: `even_rows` and `odd_rows` that separate data by even and odd positions (no physical relevance)

.. code-block:: python

  source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
  dset_source_main = VirtualDataset('source_main', source_main_data,
                                    attrs={'units': 'A', 'quantity': 'Current',
                                           'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                      'odd_rows': (slice(1, None, 2), slice(None))}
                                           })

We follow the same procedure that was followed for the `Position` datasets to create the equivalent `Spectroscopic` `Indices` and `Values` datasets:

.. code-block:: python

  source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
  source_spec_attrs = {'units': ['V', ''],
                       'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
  dset_source_spec_inds = VirtualDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                         attrs=source_spec_attrs)
  dset_source_spec_vals = VirtualDataset('SpecValues', source_spec_data, dtype=np.float16,
                                         attrs=source_spec_attrs)

With the (Virtual) datasets prepared, we can write these to a real HDF5 file using the `HDFWriter`

.. code-block:: python

  writer = HDFwriter(h5_f)
  h5_main = writer._create_dataset(h5_f, dset_source_main)
  h5_pos_inds = writer._create_dataset(h5_f, dset_source_pos_inds)
  h5_pos_vals = writer._create_dataset(h5_f, dset_source_pos_vals)
  h5_spec_inds = writer._create_dataset(h5_f, dset_source_spec_inds)
  h5_spec_vals = writer._create_dataset(h5_f, dset_source_spec_vals)

Finally, we arrive at the assertion portion of the unit test and this is the only section that will need to be changed. The following line proves that the dataset `h5_main` cannot pass the test of being a pycroscopy `Main` dataset since it has not yet been linked to the ancillary datasets

.. code-block:: python

  self.assertFalse(hdf_utils.check_if_main(h5_main))

For the example, this line could be turned into a simple print statement as:

.. code-block:: python

  print('Before linking to ancillary datasets, h5_main is a main dataset? : {}'.format(hdf_utils.check_if_main(h5_main))

This is the call to the function that we want to test:

.. code-block:: python

  # Now need to link as main!
  hdf_utils.link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

When we check to see if the dataset is now `Main`, we expect it to be true.

.. code-block:: python

  # Finally:
  self.assertTrue(hdf_utils.check_if_main(h5_main))

Again, this assertion statement can easily be turned into a print statement:

.. code-block:: python

  print('After linking to ancillary datasets, h5_main is a main dataset? : {}'.format(hdf_utils.check_if_main(h5_main))

In addition, one could also show that if a dataset is a ``Main`` dataset, we can use it as a ``Pycrodataset``. The below print statement should print the complete details regarding h5_main:

.. code-block:: python

  print(px.Pycrodataset(h5_main))

Formatting the example
----------------------
The current tool (Sphynx) requires that all examples be written in a python file formatted in a specific manner in order for the result to look like a beautiful Jupyter notebook-like documents. The code aspect of such example files is straightforward enough but here are some guidelines for formatting the text in such python scripts:

Creating Text Cells
~~~~~~~~~~~~~~~~~~~~~~
Text cells (like in Jupyter) must start with a giant line of #####

.. code-block:: python

  ####################################################################################
  # Some text here
  # Next line here.
  #
  # Empty line above signifying the end of a paragraph. Note that the previous line still
  # needs to have a '#' otherwise, the cell will be broken into two parts

Headings
~~~~~~~~~

.. code-block:: python

  ####################################################################################
  # ======================================================================================
  # H1 - Heading of the highest level
  # ======================================================================================
  # Note that the lines containing the '=' or '~' or '-' characters need to be at least as long as the text above the line
  #

  ####################################################################################
  # H2 Heading for new cell
  # ===========================
  # Conventional text below a heading - 
  #
  # You can have empty lines to signify a new paragraph
  # All this text is going to be part of a single text cell

  print('Hello World!')

  ####################################################################################
  # Text cell without any heading
  # done with text cell

  # a regular python comment
  print('Next line of code!')

Bullets
~~~~~~~

.. code-block:: python

  ####################################################################################
  # * bullet point 1
  # * bullet point 2
  
Emphasis
~~~~~~~~

.. code-block:: python

  # **some text in bold**
