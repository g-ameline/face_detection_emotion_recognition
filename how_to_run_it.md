Hey !

# intro

Those projects are usually developped and presented within notebooks;

I suggest you 3 different ways to audit the project:
- you can just read the pdf version of each notebook and the saved last output
- you can start a jupyter server and just ***read*** the notebook 
- you can build environment and rerun all the notebooks and reprocess all data   
  - (can take some computing time, depending on your machine) 
- you can run the scripts but you loose interactivity

if you want to run the scripts or the notebooks you will   
need to replicate the virtual environments in situ in specific folders
or run a contianer that will do it for you
(or install evrything globally on your machine  ... but I won't supprt that)

# about notebooks
this is some sort of interactive scripting page that run in a browser;  
- it allows you to debug and ouput/display intermediate results at any point of your script
- last output carry on in the page if copied (like here) 
- so *you* can visuzalize without rerunning everything
- there should be some pdf of the notebooks if you want to just read it
## if you want to rerun the notebook/scrip yourself:

### and you do want to install all the jupyter stuff (even python) :
go in  the repo/containing/ folders and `bash ./all_launch.sh`
then there is some printing and 
```
 To access the server, open this file in a browser:
        file:///home/mambauser/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://61f1dd9a8829:1234/lab?token=2156c4594a39c70b9e9973e5a166565976f7bf68b40a3010
        http://127.0.0.1:1234/lab?token=2156c4594a39c70b9e9973e5a166565976f7bf68b40a3010
```
try opening one of the two last links in your browser

### you want to run it locally :
- first, you will need to duplicate the virtual environment (see below)
- install kernel and stuff so that the jupyter server can run notebooks from *that* virtual environment
- get jupyter started in that repo
- run all cells in descending order (run tab then run all or shift+enter on each cells)

# duplicaing vrtual environment
this means installing ***locally*** the versionned language, packages and dependencies.
depending what you are using as package manager :  
- conda
- pip
- mamba
- other

you might want to recreate a virtual environment from :  
- environment.yml or requirement_conda.txt
  - if using conda
- requirements_pip.txt 
  - if using pip
- environment_cross.yml
  - if using macos and conda

*that's mostly stuff I read from the net, I could not test exporting my virtual environment to other package manager than conda.  
If have any issue, please tell me.*

## author
gameline
