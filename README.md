# moon_ai
Automatic grading and generation of moonboard problems.

## Description
This code implements the convolutional neural network for climbing route grade classification on the Moonboard found [here](http://cs229.stanford.edu/proj2017/final-reports/5232206.pdf).  The code for this is found in [`grade_net.py`](/grade_net.py).  [`grade_net_comp.py`](/grade_net_comp.py) implements another route climbing classification CNN with significantly more parameters.  Both algorithms were found to perform similarly.

Additionally, an auxillary classifier generative adversarial network for generation of moonboard problems is implemented in [`moon_gan.ipynb`](/moon_gan.ipynb).  This allows us to assign a grade to each generated problem produced by the generator.  An example of a generated problem with the proposed grade of 6C+ is shown below.

![Generated 6C+](/assets/generated_6C+.png)

Generated problems can be viewed through [`compose_problems.ipynb`](compose_problems.ipynb) after running the AC-GAN to generate a model.

## Installation
    $ git clone https://github.com/adamreidsmith/moon_ai
    $ cd moon_ai/
    $ sudo pip3 install -r requirements.txt
    
## Technologies Used
* Python 3
* Jupyter Notebook
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Seaborn

## License
[MIT](/LICENSE)