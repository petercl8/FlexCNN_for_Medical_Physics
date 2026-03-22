AFTER TRAINING
==============
set train_save_state/train_load_state = False (to avoid accidental overwrite)


SETUP TUNING OR TRAINING OR TESTING
===================================
user_params.py
--------------
general settings
run-type settings
QA phantoms settings

search_spaces.py (if tuning)
----------------------------
correct search space
    *including: 'stats', 'fill', and 'dropout' (IS, SI if applicable).

config net dicts (if training/testing)
--------------------------------------
correct tuned config

dataset_classes.py
------------------
correct data loading parameters

Other
-----
correct counts/Bq (switch if targets are annhilation maps)