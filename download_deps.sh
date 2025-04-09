#!/bin/bash

# download deps if not already downloaded
if [ ! -d "PromptDA" ]; then
    git clone https://github.com/Junyi42/viser.git
    git clone https://github.com/DepthAnything/PromptDA.git
fi