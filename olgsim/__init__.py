"""olgsim - synthetic patron & transaction generator for OLG-style AML and
safer-gambling detection. See simulate.py for the command-line entry point.

Dependency order (imports only flow downward):
    config, models -> reference -> factory, behaviours -> scenarios, engine -> db, calibration
"""
