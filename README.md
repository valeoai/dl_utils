# dl_utils
The library used in the Valeo Deep learning training.

Install it with
```
pip install .
```

or if you want to work on it (dev mode, will not make a copy in site-packages)
```
pip install -e .
```

Uninstall with 
```
pip uninstall dl-utils
```

Run a pep8 check with 
```
pycodestyle . --max-line-length=100
```

#### How to use:
```
from dlutils import GeneratorSingleObject
from dlutils import SGDRScheduler
from dlutils import LRFinder
from dlutils import plot_confusion_matrix
...
```
