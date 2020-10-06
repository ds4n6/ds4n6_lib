<!-- PROJECT LOGO -->

<p align="center">
  <a href="http://www.ds4n6.io">
    <img src="http://www.ds4n6.io/images/DS4N6.jpg">
  </a>

<div>
    <a href="https://twitter.com/ds4n6_io"><span class="fa fa-twitter"></span></a>
    <a href="https://www.youtube.com/channel/UC8G86XMS_b4T_hlAP4Puryg/"><span class="fa fa-youtube"></span></a>
    <a href="mailto:ds4n6@one-esecurity.com"><span class="fa fa-envelope"></span></a>
</div>
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

Visit the project page: http://www.ds4n6.io/tools/ds4n6.py.html

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

https://github.com/ds4n6/ds4n6_lib.git

### Prerequisites

The DS4N6 library works on the 3.x versions of the Python programming language. The module has external dependencies related to datascience and extraction of forensic evidence.

```
* python3 or Jupyter Notebooks
* pandas
* numpy
* matplotlib
* sklearn
* keras
```
and:
* python-evtx - EVTX export to XML - https://github.com/williballenthin/python-evtx 
```sh
    pip install python-evtx
```

### Installing

A step by step steps for installation:

1.- Clone the repo

```sh
    git clone https://github.com/ds4n6/ds4n6.git
```

And import in your python3 program o Jupyter Notebook

```python
    import DS4N6 as ds
```

## Contributing

If you think you can provide value to the Community, collaborating with Research, Blog Posts, Cheatsheets, Code, etc., contact us! 

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Jess Garcia** - *Initial work* - http://ds4n6.io/community/jess_garcia.html

See also the list of [contributors](http://ds4n6.io/community.html) who participated in this project.

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE.md](LICENSE.md) file for details
