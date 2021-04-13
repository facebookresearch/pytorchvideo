## Unit Tests


Before running the tests, please ensure that you installed the necessary additional test dependencies. 
If not installed, check the [install-README](https://github.com/facebookresearch/pytorchvideo/blob/master/INSTALL.md) on how to do it.

Use the the following command to run the tests:
```
# From root of the project
python -m unittest discover -v -s ./tests
```

To generate the coverage reports, please run the following command:
```
#Install Coverage using 
pip install coverage

# From root of the project
coverage run -m unittest discover -v -s tests
```

