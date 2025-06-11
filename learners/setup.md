Setup and Troubleshooting
ARE JupyterLab Setup for Workshop

NCI account page
Use this link to set up your NCI account
https://my.nci.org.au/ 
NCI Australian Research Environment
Make sure you use your NCI ID (eg, ab1234) and not your email address for the username.
https://are.nci.org.au/ 
NCI project folder
scratch/cd82
ARE JupyterLab session request details
Walltime (hours)  - 5
Queue - normal
Compute size - small
Project - cd82
Storage - scratch/cd82
Advanced options
Modules - python3/3.9.2
Python or Conda virtual environment base - /scratch/cd82/venv_class

When you have a Jupyter server running
Run the following code in a Jupyter notebook cell:

!mkdir -p /scratch/cd82/$USER/notebooks/data
!cp /scratch/cd82/class_wb/* /scratch/cd82/$USER/notebooks/
!cp /scratch/cd82/class_data/*.JPG /scratch/cd82/$USER/notebooks/data
!ls /scratch/cd82/$USER/notebooks/


And then use the Jupyter file browser to navigate to the directory: 	/scratch/cd82/$USER/notebooks/ (where $USER is your NCI username)
Launch Session Troubleshooting

T: My NCI login is not working

Solution: Ensure that you are entering the correct username. It should consist of 6 letters and digits, and not your email address.

T: Bad Request: Requested resource does not exist.

Solution: Reopen the https://are.nci.org.au/ page in a new tab.

T: qsub: Error: You are not a member of project cd82. You must be a member of a project to submit a job under that project.

Solution: Make you have requested access to Project cd82 via https://my.nci.org.au/ Note this may take up to 30 min to filter through the system, after you have confirmed your email address. (Check your spam and quarantine folders if you have not received an email from NCI to verify your email address.)

T: qsub: Project "cd82" does not have sufficient CPU time allocation to run this job.

Solution: Logout and login again or wait up to 30 min before you launch the job. (This error is often seen when the system is still setting up your access to the project.)

T: qsub: Error: You have not requested a project to charge for this job.

Solution: Project cd82 was not specified in resource requirement specification.

T: Failed to submit session with the following error: If this job failed to submit because of an invalid job name please ask your administrator to configure OnDemand to set the environment variable OOD_JOB_NAME_ILLEGAL_CHARS. The JupyterLab session data for this session can be accessed under the staged root directory.

Solution: Refresh the browser or switch to a new browser.
Additional Resources
Upcoming QCIF-NCI Workshops
Introduction to the Unix Shell (Jun 2025) - NCI 16 June
Version Control with Git (Jun 2025) - NCI 26 June
Decision Trees and Ensemble Methods in Machine Learning (Jul 2025) - NCI 16 July - SOLD OUT
Clustering and Unsupervised Methods in Machine Learning (Jul 2025) - NCI 29 July - SOLD OUT
Introduction to High Performance Computing (HPC) (Aug 2025) - NCI - 14 August
Introduction to Image Classification (Aug 2025) - NCI - 19 August
NCI Resources
NCI Australia - https://nci.org.au/
NCI's Public Documentation Home - https://opus.nci.org.au/
Gadi User Guide - https://opus.nci.org.au/spaces/Help/pages/236880325/Gadi+User+Guide
ARE User Guide - https://opus.nci.org.au/spaces/Help/pages/162431120/ARE+User+Guide
JupyterLab on ARE - https://opus.nci.org.au/spaces/Help/pages/163250554/3.+JupyterLab+App
NCI Help Desk - https://nci.org.au/users/nci-helpdesk
2025 Training and Education Events Calendar - https://opus.nci.org.au/spaces/Help/pages/48497461/NCI+Training+and+Educational+Events
We particularly recommend Introduction to Unix and Introduction to HPC for working confidently with Gadi, regardless of the programming language you use.
JupyterLab Resources
The Jupyter Project - https://jupyter.org/
JupyterLab User Guide - https://jupyterlab.readthedocs.io/en/stable/user/index.html





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

