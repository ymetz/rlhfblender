"""
    Predict uncertainty for a given list of states
    To achieve this:

    (1) Load reward model and policy from given checkpoint
    (2) For each state in list, predict action with policy
    (3) For each state, predict reward with reward model
    (4) Save predictions to session_dir
"""