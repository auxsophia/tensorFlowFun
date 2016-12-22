# tensorFlowFun
Random TensorFlow (machine learning framework) projects for fun and to learn.


## blackecho
Blackecho is the username of an excellent programmer who released ready to use [TensorFlow models](https://github.com/blackecho/Deep-Learning-TensorFlow). The webstie for the documentation can be seen [here](http://deep-learning-tensorflow.readthedocs.io/en/latest/).

### Note: In order to clone the blackecho folder along with tensorFlowFun, you must use the following command:
```
git clone --recursive https://github.com/auxsophia/tensorFlowFun.git
```

If you cloned normally and missed the blackecho folder or just wish to update the submodule:
```
git submodule update --init --recursive
```
The submodule won't update on its own, so we may need to run this manually when it does.

### Known error:
Encountered when trying to run the deep autoencoder on cifar10, import error for yadlt.models.rbm_models.
I ran the following in the yadlt/models/ directory
```
export PYTHONPATH=.
```
and it worked nicely.
