AFTER TRAINING
==============
set run_mode != train (to avoid overwriting model)

SETUP TUNING OR TRAINING OR TESTING
===================================
user_params.py
--------------
general settings
run-type settings
(if altering data) QA phantoms settings


search_spaces.py (if tuning)
----------------------------
correct search space
    *including: 'stats', 'fill', and 'dropout' (IS, SI if applicable).


config net dicts (if training/testing)
--------------------------------------
correct tuned config


dataset_classes.py
------------------
(if altering data) correct data loading parameters


Other
-----
correct counts/Bq (switch if targets are annhilation maps)