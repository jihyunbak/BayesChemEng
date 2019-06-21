BayesChemEng
============


Matlab code for an effective inference method for analyzing costly experiments, often with multi-dimensional parameters, performed at multiple different settings (sets of design variables) and few repetitions.

Repository is maintained by Ji Hyun Bak (jhbak@kias.re.kr).


#### Reference:

* Na J\*, Park S\*, Bak JH, Kim M, Lee D, Yoo Y, Kim I, Park J, Lee U, & Lee JM (2019). Bayesian inference of aqueous mineral carbonation kinetics for carbon capture and utilization. 
_Industrial & Engineering Chemistry Research_. 
[(link)](https://doi.org/10.1021/acs.iecr.9b01062)

If you are only interested in the algorithm, the [document](doc/doc_alg.pdf) in this repository contains the relevant part of the supplementary material that is published with the main paper.


## Setup

### Obtaining

You can do one of the following to obtain the latest code package.

* **Download**:   zipped archive  [BayesChemEng-master.zip](https://github.com/jihyunbak/BayesChemEng/archive/master.zip)
* **Clone**: clone the repository from github: ```git clone https://github.com/jihyunbak/BayesChemEng.git```

### Running

* You need MATLAB with the {statistics and machine learning, neural network, curve fitting} toolboxes.
* Code was developed using MATLAB R2016b, although older versions may also work. Please let us know if you have any issue with older releases.



## Example scripts

We provide demo scripts to illustrate how the algorithm works.

### Single dataset

* See `demo1_single.m` for a running example of iterative sampling algorithm, applied to a single dataset.

### Multiple datasets

* See `demo2_multi.m` for a similar example, but applied to multiple datasets. 
In this case, there is one set of (hidden) parameters that defines the process being observed, 
and two "experiments" are performed at two different design variables. 
As you can see in the demo script, 
our BayesChemEng package simply takes the multi-dataset collection as the input, 
along with a set of interpretable options, 
and takes care of the rest of posterior inference.

