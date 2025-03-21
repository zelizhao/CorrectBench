# Connecting to a Server via VPN and VSCode Remote-SSH

This guide will walk you through the process of connecting to a remote server using a VPN and Visual Studio Code with the Remote-SSH extension.

## Prerequisites

Before you begin, ensure you have the following:

- Visual Studio Code installed on your local machine
- The Remote-SSH extension for Visual Studio Code
- VPN client software to establish a secure connection

## Step 1: Download and Install VPN

First, you need to download and install the VPN client. Use the following link to download the VPN software:

**VPN Download Address:** [VPN Download Link Here](https://download.yuekeo.com:8443/Anyconnet/)

## Step 2: Configure VPN Connection

Once the VPN client is installed, configure it with the following details:

- **VPN Connection Address:** ykssl.yuekeo.com:5443
- **VPN Group:** shengine
- **VPN Username:** squirrel01
- **VPN Password:** 2a3uws68

Follow the VPN client's instructions to establish a connection to the VPN.

## Step 3: Connect to the Server Using VSCode Remote-SSH

After you are connected to the VPN, proceed with the following steps to connect to the server using VSCode:

1. Open Visual Studio Code.
2. Press `Ctrl+Shift+P` to open the Command Palette.
3. Type `Remote-SSH: Add New SSH Host` and select it.
4. You will be prompted to enter the SSH connection command. Use the following format:

```bash
ssh -p 10026  yueke@58.57.119.83
```
5. After entering the command, the system may prompt you to enter the password for the SSH connection. Use the following password:
**SSH Password:** lehigh2025

6. Once authenticated, VSCode will connect to the remote server, and you will be able to work on your projects as if you were working locally.
# SelfRefine.py README

## Overview
`selfrefine.py` is a Python script located in the `/home/yueke/correct/hcrselfcorrection/Self-Correction-Benchmark/method` directory. This script is designed for the Self-Correction-Benchmark project and performs a set of specific operations. Please follow the instructions below to ensure the script runs correctly.

## Prerequisites
Before running `selfrefine.py`, make sure to meet the following prerequisites:

- You have activate a Conda environment named `Intrinsic`.

## Running the Script
To execute `selfrefine.py`, carefully follow these steps:

1. **Accessing the Directory**:
Before you begin, ensure you are in the correct directory:

```bash
cd /home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/method
```
2. **Activate the Conda Environment**:
   It is crucial to activate the Conda environment `Intrinsic` before running the script. This ensures that all necessary dependencies and environment variables are set correctly. Use the following command in your terminal:

```bash
conda activate Intrinsic
```

3. **Run the Script**:
After activating the Conda environment, navigate to the directory containing `selfrefine.py` and run the script using Python:

```bash
python selfrefine.py
```

The script will start executing, and you may see various output messages in the terminal.

## Output and Warnings
During the execution of `selfrefine.py`, you might encounter the following information:

```bash
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input’s attention_mask to obtain reliable results. Setting pad_token_id to eos_token_id:128001 for open-end generation.
```
This message indicates that the model is being initialized and is a normal part of the script's operation. You can safely ignore it and take no action.

## Modification Policy

Please note that modifications to the codebase are restricted to the following directories:

- `/home/yueke/correct/hcrselfcorrection/dataset`
- `/home/yueke/correct/hcrselfcorrection/Self-Correction-Benchmark/task`