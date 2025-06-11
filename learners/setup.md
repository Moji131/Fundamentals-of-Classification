Image Classification with Convolutional Neural Networks: Summary and Setup MathJax.Hub.Config({ config: \["MMLorHTML.js"\], jax: \["input/TeX","input/MathML","output/HTML-CSS","output/NativeMML", "output/PreviewHTML"\], extensions: \["tex2jax.js","mml2jax.js","MathMenu.js","MathZoom.js", "fast-preview.js", "AssistiveMML.js", "a11y/accessibility-menu.js"\], TeX: { extensions: \["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"\] }, tex2jax: { inlineMath: \[\['\\\\(', '\\\\)'\]\], displayMath: \[ \['$$','$$'\], \['\\\\\[', '\\\\\]'\] \], processEscapes: true } }); 

[Skip to main content](#main-content)

![Carpentries Incubator](assets/images/incubator-logo.svg) [Pre-Alpha](https://docs.carpentries.org/resources/curriculum/lesson-life-cycle.html) This lesson is in the pre-alpha phase, which means that it is in early development, but has not yet been taught.

*   *   Light
    *   Dark
    *   Auto

Learner View

*   Instructor View

* * *

Menu

![Carpentries Incubator](assets/images/incubator-logo-sm.svg)

Image Classification with Convolutional Neural Networks

*   Image Classification with Convolutional Neural Networks
*   [Key Points](key-points.html)
*   [Glossary](reference.html#glossary)
*   [Learner Profiles](profiles.html)
*   More
    *   [Reference](reference.html)

[Search the All In One page](aio.html)

Image Classification with Convolutional Neural Networks

%

*   Toggle Theme
    *   Light
    *   Dark
    *   Auto

Learner View
------------

[Instructor View](instructor/index.html)

* * *

EPISODES
--------

Current Chapter Summary and Setup

*   [NCI Account Setup](#nci-account-setup)
*   [NCI Australian Research Environment (ARE)](#nci-australian-research-environment-are)
*   [Getting the Data](#getting-the-data)

[1\. Running and Quitting JupyterLab on NCI](00-run-quit.html)

[2\. Introduction to Deep Learning](01-introduction.html)

[3\. Introduction to Image Data](02-image-data.html)

[4\. Build a Convolutional Neural Network](03-build-cnn.html)

[5\. Compile and Train (Fit) a Convolutional Neural Network](04-fit-cnn.html)

[6\. Evaluate a Convolutional Neural Network and Make Predictions (Classifications)](05-evaluate-predict-cnn.html)

[7\. Share a Convolutional Neural Network and Next Steps](06-share-cnn-next-steps.html)

* * *

RESOURCES
---------

*   [Key Points](key-points.html)
*   [Glossary](reference.html#glossary)
*   [Learner Profiles](profiles.html)
*   [Reference](reference.html)

* * *

[See all in one page](aio.html)

* * *

[Next](00-run-quit.html)

[Next: Running and Quitting...](00-run-quit.html)

* * *

Summary and Setup
=================

This is a new lesson built with [The Carpentries Workbench](https://carpentries.github.io/sandpaper-docs).

This lesson is designed for Software Carpentry users who have completed [Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder/) and want to jump straight into image classification. We recognize this jump is quite large and have done our best to provide the content and code to perform these types of analyses.

The NCI-QCIF Training Partnership Project version of this lesson uses python virtual environments to run Jupyter Notebooks on [NCI’s Gadi supercomputer](https://nci.org.au/news-events/events/introduction-gadi-4).

It uses the [TensorFlow](https://www.tensorflow.org/) software library in a **CPU** only environment.

### Callout

Please note this lesson is designed to work with CPU only environments. This was an intentional decision to avoid the difficulties in setting up GPU environments. If you are an advanced user and choose to set up a GPU environment, you are on your own. We will not be able to troubleshoot any issues with GPU set up on the day of the workshop.

NCI Account Setup[](#nci-account-setup)
---------------------------------------

* * *

Sign up for an [NCI account](https://my.nci.org.au) if you don’t already have one.

Select **Projects and groups** from the left hand side menu and then select the **Find project or group** tab. Search for **cd82**, the NCI-QCIF Training Partnership Project, and ask to join.

![NCI Find a project or group page](fig/0_my_nci_project_cd82.png)

NCI Australian Research Environment (ARE)[](#nci-australian-research-environment-are)
-------------------------------------------------------------------------------------

* * *

Connect to [NCI Australian Research Environment](https://are.nci.org.au).

Be sure you use your NCI ID (eg, ab1234) for the username and not your email address.

Under **Featured Apps**, find and click the **JupterLab: Start a JupyterLab instance** option.

![NCI ARE JupyterLab](fig/0_nci_are_mainpage.png)

To Launch a JuptyerLab session, set these resource requirements:

Resource

Value

Walltime (hours)

5

Queue

normal

Compute Size

small

Project

cd82

Storage

scratch/cd82

**Advanced Options…**

Modules

python3/3.9.2

Python or Conda virtual environment base

/scratch/cd82/venv\_icwcnn

Then click the Launch button.

This will take you to your interactive session page you will see that that your JupyterLab session is Queued while ARE is searching for a compute node that will satisfy your requirements.

Once found, the page will update with a button that you can click to **Open JupyterLab**.

Here is a screenshot of a JupyterLab landing page that should be similar to the one that opens in your web browser after starting the JupyterLab server on either macOS or Windows.

![JupyterLab landing page](fig/0_jupyterlab_landing_page.png)

Getting the Data[](#getting-the-data)
-------------------------------------

* * *

This lesson uses the CIFAR-10 image dataset that comes prepackaged with Keras. There are no additional steps needed to access the data.

[Next](00-run-quit.html)

[Next: Running and Quitting...](00-run-quit.html)

* * *

This lesson is subject to the [Code of Conduct](CODE_OF_CONDUCT.html)

[Edit on GitHub](https://github.com/erinmgraham/icwithcnn/edit/main/index.md) | [Contributing](https://github.com/erinmgraham/icwithcnn/blob/main/CONTRIBUTING.md) | [Source](https://github.com/erinmgraham/icwithcnn/)

[Cite](https://github.com/erinmgraham/icwithcnn/blob/main/CITATION) | [Contact](/cdn-cgi/l/email-protection#2247504b4c0c4550434a434f624841570c4746570c4357) | [About](https://carpentries.org/about/)

Materials licensed under [CC-BY 4.0](LICENSE.html) by the authors

Template licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by [The Carpentries](https://carpentries.org/)

Built with [sandpaper (0.16.12)](https://github.com/carpentries/sandpaper/tree/0.16.12), [pegboard (0.7.9)](https://github.com/carpentries/pegboard/tree/0.7.9), and [varnish (1.0.6)](https://github.com/carpentries/varnish/tree/1.0.6)

[  
Back To Top](#top)

{ "@context": "https://schema.org", "@type": "LearningResource", "@id": "https://erinmgraham.github.io/icwithcnn/index.html", "inLanguage": "en", "dct:conformsTo": "https://bioschemas.org/profiles/LearningResource/1.0-RELEASE", "description": "A Carpentries Lesson teaching foundational data and coding skills to researchers worldwide", "keywords": "software, data, lesson, The Carpentries", "name": "Image Classification with Convolutional Neural Networks", "creativeWorkStatus": "active", "url": "https://erinmgraham.github.io/icwithcnn/index.html", "identifier": "https://erinmgraham.github.io/icwithcnn/index.html", "dateCreated": "2023-05-03", "dateModified": "2025-06-10", "datePublished": "2025-06-10" } feather.replace();



<!-- Collect your link references at the bottom of your document -->

[Plotting and Programming in Python]: https://swcarpentry.github.io/python-novice-gapminder/
[Conda]: https://docs.conda.io/projects/conda/en/latest/
[Python]: https://python.org
[Anaconda]: https://www.anaconda.com/products/individual
[Windows - Video tutorial]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[Mac OS X - Video tutorial]: https://www.youtube.com/watch?v=TcSAln46u9U
[these instructions]: https://docs.anaconda.com/anaconda/install/update-version/
[pip]: (https://pip.pypa.io/en/stable/)
[Spyder]: https://www.spyder-ide.org/
[scripts, files, and model outputs]: https://drive.google.com/file/d/1SpcusVYomhukFKWuUcK7LwF7RtrKB8Z_/view?usp=drive_link

