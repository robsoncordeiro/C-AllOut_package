# The Method C-AllOut for Catching and Calling Outliers by Type

C-AllOut is a parameter-free, distance-based, powered by Slim-tree, scalable method for outlier detection and outlier type annotation.

# Main Sections
1. [Directory Tree](#Directory_Tree_13)
2. [C-AllOut Usage](#CAllOut_Usage_53)
3. [Synthetic Dataset Generator](#Synthetic_Dataset_Generator_128)
4. [Realistic Dataset Generator](#Realistic_Dataset_Generator_170)
5. [C-AllOut Competitors](#CAllOut_Competitors_221)

## Directory Tree

A summary of the file structure can be checked in the following directory tree. Every directory purpose is described in a short comment next to its name.

```bash
Callout
├───code
│   ├───docs \\ Slim-tree documentation
│   │   └───image
│   ├───include \\ Slim-tree main code
│   │   └───arboretum
│   ├───lib \\ Slim-tree static libraries
│   │   ├───bcc5
│   │   ├───bcc6
│   │   └───gcc
│   ├───sample \\ Apps that instantiate Slim-tree
│   │   └───callout \\ C-AllOut instantiation of Slim-tree and overall code
│   │       ├───benchmark_datasets \\ For convenience, benchmark datasets where needed
│   │       ├───obj_level \\ Tree-level of the objects in the tree
│   │       ├───progression_stats \\ AUCPR and AUCROC for each iteration
│   │       ├───realistic_datasets \\ For convenience, realistic datasets where needed
│   │       ├───scores \\ Outlierness measures as given by Slim-tree
│   │       ├───stats \\ Final AUCPR and AUCROC performance results
│   │       └───synthetic_datasets \\ For convenience, synthetic datasets where needed
│   └───src \\ Slim-tree low-level API code
├───competitors \\ C-AllOut competitors implementation and instantiation
└───testbeds \\ The three testbeds discussed in the paper Evaluation section
    ├───testbed_1_benchmarks \\ Testbed 1: benchmark datasets
    │   └───original_data
    ├───testbed_2_synthetic \\ Testbed 2: synthetic datasets
    │   ├───S1 \\ Files related to dataset S1
    │   ├───S2 \\ Files related to dataset S2
    │   └───S3 \\ Files related to dataset S3
    └───testbed_3_realistic \\ Testbed 3: realistic datasets
        ├───3d_datasets_figures \\ 3D plots for the realistic 3D datasets
        ├───final_preprocessed_datasets \\ Datasets as input to C-AllOut
        ├───generator_output \\ Datasets and metadata as created by the generator
        └───original_datasets \\ Original datasets used as input for the generator
```

## C-AllOut Usage

### Dependencies

C-AllOut and its competitors use a number of open source tools to work properly. The following packages should be installed before starting:

#### Related to C++

- make - Unix tool for running Makefiles.
- gcc/g++ - Compiler for C++ code.

#### Related to Python
- [Python] - Python interpreter in version 3.7 or above;
- [numpy] - Math package for Python;
- [matplotlib] - Tool for creating dataset plots;
- [sklearn] - Machine learning library for Python. Implementation for competitor _kNN_ is from this package;
- [kneed] - Algorithm for finding the knee/elbow point of a line;
- [PyOD] - Python toolkit for detecting outlying objects. Implementations for competitors _OCSVM_ and _LOF_ are from this package.

### Installation

All C++ dependencies should be already installed and available if you have any Unix-like OS. For Windows users, **Windows Subsystem for Linux (WSL)**, **cygwin** and **minGW** can provide the C++ dependencies.

All Python dependencies can be installed via the usual pip command:

```sh
pip install package-name
```

### Building C-AllOut

Go into path _code/callout/lib/gcc_ and run with no options:

```sh
make
```

This will build the static library used by Slim-tree.

Next step is to compile C-AllOut's instance of Slim-tree. In path _code/callout/callout_src/callout_ run with no options:

```sh
make
```

Note that the Slim-tree code related to C-AllOut can be found in file _code/include/arboretum/stSlimTree.cc_, in function _CalloutOutlierness_ and its auxiliary functions.

### Running C-AllOut

Go to the path where C++ executable file "callout" is installed. Use the following command to run the python script used to generate the Testbed 1 results available in the paper (no parameter input needed):

```sh
python benchmarks.py
```

The python script will run C-AllOut with 10 random permutations for each dataset. Two _.csv_ files will be output for every dataset:
- _pxr.csv_ files have results for AUCPR;
- _roc.csv_ files have results for AUCROC.

The experiments for Testbed 3 can be reproduced using another Python script which deals with all the rankings and not only the overall ranking:

```sh
python benchmarks_realistic.py
```

For Testbed 2, the same Python script can be used. The only change needed is in line 12, to make the script use the synthetic list of datasets instead of the realistic one.

### Generating the Relative AP Gain Figure

C-AllOut has a iteration mechanism in which sequential trees are created. For each tree, the input instances are inserted into the tree using a different ordering. The objective is to insert outlier instances for last. This way, we are able to improve C-AllOut's performance. After a couple of iterations there is a convergence in AUCPR and AUCROC performance. More details can be found in the paper. To generate the figure in the paper, which has the AP average gain against the first iteration, the following script can be run in path _code/callout/callout_src/callout_:

```sh
python iterative_gains_graph.py
```

## Synthetic Dataset Generator

For creating datasets *S1*, *S2* and *S3*, as well as their variations, we developed a synthetic data generator. The generator is completely implemented in python.

### Dependencies

- [Python] - Python interpreter in version 3.7 or above;
- [numpy] - Scientific computing package for Python;
- [matplotlib] - Tool for creating dataset plots;
- [sklearn] - Machine learning library for Python;
- [random] - Pseudo-random number generator.

### Running the Synthetic Dataset Generator

Go to path _testbeds/testbed_2_synthetic/_ and run the following command:

```sh
python synthetic_generator.py
```

Four variations for a single dataset will be generated in accordance to the parameters provided, where X is the dataset name:

- X_A: dataset having all three outliers types;
- X_L: dataset having only local outliers;
- X_G: dataset having only global outliers;
- X_C: dataset having only collective outliers.

Each variation will be created along with two label files. Taking X_A for instance:

- y_X_A: is a label file with two possible labels, 0 for inlier and 1 for outlier.
- ytype_X_A: is a label file with four possible labels, 0 for inlier, 1 for local outlier, 2 for global outlier and 3 for collective outlier.

To have exactly the same synthetic datasets used in the Evaluation section of our paper (the synthetic datasets available in our repository), one must run the preprocessing script:

```sh
python preprocessing_synthetic.py
```

This script removes duplicate instances, scales data to 0 mean and 1 standard deviation, adds IDs for each instance and shuffle all instances.

More details for the synthetic dataset generator can be found in its directory _testbeds/testbed_2_synthetic_.

## Realistic Dataset Generator

Datasets *ALOI*, *Glass*, *skin*, *smtp* and *Waveform*, as well as their variations, originate from a realistic dataset generator. Our realistic dataset generator is a modified version of the generator provided by (Steinbuss and Böhm, 2021) that is implemented using *R* language. Our modification basically adds instructions for saving the generated datasets to disk and edits some lines of code so that we only get the datasets we need and not every dataset it originally creates. The generator takes as input the original version of a dataset and fits a GMM to the input data. The original datasets are all available in the zipped directory and have the same names as our generated datasets (ALOI, Glass, ...). The original, unmodified, version of the generator has trouble working with some specific input datasets, not used in our experiments. So, do not expect the original or our modified version to work flawlessly.

### Dependencies

#### Related to R
The generator proposed in (Steinbuss and Böhm, 2021) has many dependencies which can be quickly installed using a script provided along with the generator. To install all dependencies unzip "realistic_generator.zip" and follow the instructions provided in the "README.html" file available in the unzipped directory.

#### Related to Python
- [Python] - Python interpreter in version 3.7 or above;
- [numpy] - Scientific computing package for Python;
- [sklearn] - Machine learning library for Python;
- [random] - Pseudo-random number generator.

### Running the Realistic Dataset Generator

Go to path _testbeds/testbed_3_realistic/realistic_generator_ and follow the instructions available in the "README.html" file until section "Setting up Batchtools". Note that the list of datasets to be used as input is defined in the "design.R" script. Having installed all dependencies successfully, run the following command:

```sh
Rscript run.R
```

This command will set up the realistic dataset generator and run it. Make sure that the R interpreter has writing priviledges to path _testbeds/testbed_3_realistic/realistic_generator_. After that, _testbeds/testbed_3_realistic/realistic_generator/gendatasets_ should be populated with files for the generated dataset. The generator itself only creates local and global outliers and is not able to provide outlier type labels. To solve these issues we have developed a python script that works on the output files of the realistic generator. By using the densities of each point, provided by the generator, the script labels a desired number of points with the smallest densities as global outliers. Another desired number of points with the largest densities are labeled local outliers. The script also creates collective outliers by generating gaussian micro-clusters around a random selection of the global outliers. Thus, this script is a crucial step to get outlier annotation datasets. It should be placed and executed in the same path the generator output files are (_testbeds/testbed_3_realistic/generator_output_).

```sh
python post_generator_adjustments.py
```

Four variations for a single dataset will be generated in accordance to the parameters provided, where S is the dataset name:

- X_A: dataset having all three outliers types;
- X_L: dataset having only local outliers;
- X_G: dataset having only global outliers;
- X_C: dataset having only collective outliers.

Each variation will be created along with two label files. Taking X_A for instance:

- y_X_A: is a label file with two possible labels, 0 for inlier and 1 for outlier.
- ytype_X_A: is a label file with four possible labels, 0 for inlier, 1 for local outlier, 2 for global outlier and 3 for collective outlier.

To have exactly the same realistic datasets used in the Evaluation section of our paper (the realistic datasets available in our repository), one must run the preprocessing script:

```sh
python preprocessing_realistic.py
```

This script removes duplicate instances, adds IDs for each instance and shuffle all instances. Data is not going to be scaled to 0 mean and 1 standard deviation as this is already done in a preprocessing step of the realistic generator.

More details for the realistic dataset generator can be found in its directory _testbeds/testbed_3_realistic_

## C-AllOut Competitors

The competitors used in our Evaluation section can be found in the directory named _competitors_. Note that, to run these scripts, they should be placed in the same directory that has the preprocessed datasets to be used, i.e. _testbeds/testbed_3_realistic/final_preprocessed_datasets_. Auxiliary scripts, _datasets_list.py_ and _experiments_utils.py_, should also be placed in the same directory.

The script named _isolation_simforest.py_ is an adaptation of the method proposed by (Czekalski and Morzy, 2021) that is available at [SIF]. _sra.py_ is our own implementation of the method proposed by (Nian et al., 2016), following the paper description of the algorithm as close as possible.

The following command will run all of the competitors for Testbed 1. Namely, LOF, kNN, OCSVM and SIF.

```sh
python benchmark_competitors.py
```

Script _sra.py_ can be used to reach competitors results for Testbed 2 and 3. We do not provide the square similarity matrices files used as input by SRA since some of those files can grow up to more than 50 GBs. However, they can be easily obtained by calculating the dot product between every pair of instances in the dataset.

_This software was designed in Unix-like systems, it is not yet fully tested in other OS._

[//]: #
   [python]: <https://www.python.org/downloads/release/python-370>
   [numpy]: <https://numpy.org/doc/stable/index.html>
   [matplotlib]: <https://matplotlib.org>
   [sklearn]: <https://scikit-learn.org/stable>
   [kneed]: <https://kneed.readthedocs.io/en/stable>
   [pyod]: <https://pyod.readthedocs.io/en/latest>
   [random]: <https://docs.python.org/3/library/random.html>
   [SIF]: <https://github.com/sfczekalski/similarity_forest/blob/master/simforest/outliers/isolation_simforest.py>