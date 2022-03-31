## The libraries required to run the project including the full version of each library

numpy==1.20.1
opencv-python==4.5.3.56

## How to run each task and where to look for the output file.

In config.py, change PATH_TESTS to the path (relative to main.py or absolute) to the path that will contain two folders: clasic and jigsaw. Those folders will contain only jpg files that will have the filename of form '01.jpg', '07.jpg', '10.jpg', '100.jpg'...

Solutions will be written in evaluare/fisiere_solutie/Avram_Daniel_334. The path can be changed in config.py, if it's needed.

```src``` and ```evaluare``` must remain in the same directory!

From the ```src``` directory:

# Task 1: 
script without bonus:   ```python main.py --type clasic```
script with bonus:      ```python main.py --type clasic --bonus```

In both cases, the solutions will be saved in the same folder (specified in config.py), but with different sufixes.

# Task 2:
script without bonus:   ```python main.py --type jigsaw```
script with bonus:      ```python main.py --type jigsaw --bonus```

In both cases, the solutions will be saved in the same folder (specified in config.py), but with different sufixes.

## For evaluation
Don't forget to chage the path to ground truths in evalueaza_solutie.py. Run evalueaza_solutie.py from ```evaluare``` directory!