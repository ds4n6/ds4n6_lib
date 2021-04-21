<!-- PROJECT LOGO -->

<p align="center">
  <a href="http://www.ds4n6.io">
    <img src="http://www.ds4n6.io/images/DS4N6.jpg">
  </a>
</p>

<a href="http://www.ds4n6.io" title=""><img src="http://ds4n6.io/images/logo-s.png" alt="" /></a>

DS4N6 stands for Data Science Forensics.

We also refer to this project as DSDFIR, AI4N6 or AIDFIR, since Data Science (DS) includes Artificial Intelligence (AI), and the project goes beyond the strictly Forensics, covering the whole Digital Forensics & Incident Response (DFIR) discipline (and sometimes even beyond). But hey, we had to give the project a catchy name!

The Mission of the DS4N6 project is simple:

```
Bringing Data Science & Artificial Intelligence
to the fingertips of the average Forensicator,
and promote advances in the field
```

The first (modest) alpha version of our ds4n6 python library, together with some easy-to-use python scripts, was originally made public after the presentation at the SANS DFIR Summit US, July 16-17.
**For detailed information about the Project, the Library, its Functions, its Usage, etc., visit the project page: http://www.ds4n6.io/tools/ds4n6.py.html**

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

https://github.com/ds4n6/ds4n6_lib.git

### Prerequisites

The DS4N6 library works on the 3.x versions of the Python programming language. The module has external dependencies related to datascience and extraction of forensic evidence.

Install requirements:

    - python-evtx
    - Evtx
    - ipyaggrid
    - IPython
    - ipywidgets
    - keras
    - matplotlib
    - nbformat
    - numpy
    - pandas
    - pyparsing
    - qgrid
    - ruamel.yaml
    - sklearn
    - tensorflow
    - tqdm
    - traitlets
    - xmltodict

### Installation

The installation can be easily done through pip.

#### pip installation

```sh
    pip install python-evtx Evtx ipyaggrid IPython ipywidgets keras matplotlib nbformat numpy pandas pyparsing qgrid ruamel.yaml sklearn tensorflow tqdm traitlets xmltodict ds4n6-lib
```

Finally, import in your python3 program or Jupyter Notebook as "ds".

```python
    import ds4n6_lib as ds
```

## Contributing

If you think you can provide value to the Community, collaborating with Research, Blog Posts, Cheatsheets, Code, etc., contact us!

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

### download from github

All you will need to do is to clone the library, install the test, create a virtual enviroment to use it and active it.

```sh
    
    git clone https://github.com/ds4n6/ds4n6_lib    

    virtualenv -p python3.7 .test
    source .test/bin/activate
    
    pip install -r requirements.txt 
```

## Authors

* **Jess Garcia** - *Initial work* - http://ds4n6.io/community/jess_garcia.html

See also the list of [contributors](http://ds4n6.io/community.html) who participated in this project.

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE](LICENSE) file for details
