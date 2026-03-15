AFTER TRAINING
==============
set train_load_state = True (to avoid accidental overwrite)



SETUP TUNING OR TRAINING OR TESTING
===================================
user_params.py
--------------
general settings
run-type settings
QA phantoms settings

search_spaces.py (if tuning)
----------------------------
correct counts/Bq (switch if targets are annhilation maps)
correct search space (including 'stats' and 'fill' options)

config net dicts (if training/testing)
--------------------------------------
correct tuned config

dataset_classes.py
------------------
correct data loading parameters