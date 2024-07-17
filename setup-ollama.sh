#!/usr/bin/env bash

ollama serve &
ollama pull llama3
ollama pull mxbai-embed-large