# -*- coding: utf-8 -*-

#------------------------- \cond COPYRIGHT --------------------------#
#                                                                    #
# Copyright (C) 2026 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#----------------------------- \endcond -----------------------------#


import sys

## A small helper class to evaluate key value pairs separated by "=" from command line parameters, like for example:
##   python myprogram.py key=value
class CommandLineParser:

    ## Create a new CommandLineParser instance.
    ## \param argv: command line arguments from sys.argv
    def __init__(self, argv=None):
        self._values = {}
        self._flags = set()
        self._path = ""
        self._has_path = False

        if argv is None:
            argv = sys.argv

        argc = len(argv)
        for i in range(1, argc):
            arg = argv[i]
            if arg is None:
                continue
            self._add_token(arg, i, argc)

    ## Checks if the given \p key is available.
    ## \param key The key to check.
    ## \return Returns True if the given \p key was found in command line parameters.
    def has(self, key):
        return key in self._values

    ## Checks if the given \p flag is available.
    ## \param name Name of the flag to check.
    ## \return True if the given \p flag was found in command line parameters.
    def hasFlag(self, name):
        if len(name) >= 2 and name[:2] == "--":
            return name[2:] in self._flags
        return name in self._flags

    ## Checks if we have the first or last parameter not being a key=value pair but a path.
    ## \return True if the first or last param is a string not consisting of a key=value pair.
    def hasPath(self):
        return self._has_path

    ## Returns the first or last parameter if it is a path like string parameter, see \ref hasPath().
    ## \param default_value The value to be returned when there is no path-like parameter available. Defaults to empty string, not None.
    ## \return The string of tha path-like parameter.
    def getPath(self, default_value=""):
        if not self._has_path:
            return default_value
        return self._path

    ## Return the value for a given key as string. Default value will be returned if key was not given in parameters.
    ## \param key The key to get the value for.
    ## \param default_value The value to return in case of key was not given in command line parameters. Defaults to empty string, not None.
    def get(self, key, default_value=""):
        return self._values.get(key, default_value)

    ## Get the value of a key as an integer value.
    ## \param key The key to get the value for.
    ## \param default_value The value to return in case of key was not given in command line parameters. Defaults to 0.
    def getInt(self, key, default_value=0):
        if key not in self._values:
            return default_value
        try:
            return int(self._values[key])
        except ValueError:
            return default_value

    ## Get the value of a key as a floating point value.
    ## \param key The key to get the value for.
    ## \param default_value The value to return in case of key was not given in command line parameters. Defaults to 0.0.
    def getFloat(self, key, default_value=0.0):
        if key not in self._values:
            return default_value
        try:
            return float(self._values[key])
        except ValueError:
            return default_value

    ## Internal helper function.
    def _add_token(self, token, index, argc):
        if len(token) >= 3 and token[:2] == "--":
            name = token[2:]
            if name:
                self._flags.add(name)
            return

        pos = token.find("=")
        if pos == -1 or pos == 0:
            if not self._has_path and (index == 1 or index == argc - 1):
                self._path = token
                self._has_path = True
            return

        key = token[:pos]
        value = token[pos + 1:]
        self._values[key] = value
