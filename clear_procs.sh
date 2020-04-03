#!/bin/bash
kill -9 `ps -ef | grep DeepMimic | tr ' ' ':' | cut -d':' -f2`
