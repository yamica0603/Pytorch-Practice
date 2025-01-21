{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "02edc450",
   "metadata": {
    "papermill": {
     "duration": 0.004594,
     "end_time": "2025-01-21T14:04:24.046567",
     "exception": false,
     "start_time": "2025-01-21T14:04:24.041973",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Pytorch\n",
    " * Tensors is the fundamental data structure in pytorch.\n",
    " * it is an array which performs mathematical operations and is the building block of neural networks\n",
    " * can be created form python **lists** using **tensor.torch(list_name)** or from **numpy** array using toch.fro_numpy(**list_name**)\n",
    " * similar to numpyarrays tensors are **multidimensional** representations of their elements.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "aac29291",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:24.056086Z",
     "iopub.status.busy": "2025-01-21T14:04:24.055641Z",
     "iopub.status.idle": "2025-01-21T14:04:27.959812Z",
     "shell.execute_reply": "2025-01-21T14:04:27.958481Z"
    },
    "papermill": {
     "duration": 3.911319,
     "end_time": "2025-01-21T14:04:27.962052",
     "exception": false,
     "start_time": "2025-01-21T14:04:24.050733",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "list =  [[1,2,3],[4,5,6]]\n",
    "tensor = torch.tensor(list) #creation of tensors using lists \n",
    "\n",
    "list1 = [[7,8,9],[1,2,3]]\n",
    "tensor1 = torch.tensor(list1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50a028e8",
   "metadata": {
    "papermill": {
     "duration": 0.004095,
     "end_time": "2025-01-21T14:04:27.970630",
     "exception": false,
     "start_time": "2025-01-21T14:04:27.966535",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#Tensor attributes\n",
    ">* Tensor shape - var_name.shape\n",
    ">* Tensor datatype - var_name.dtype\n",
    ">* check which devce the tensor is loaded on (cpu/gpu) - var_name.device\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "efbc1cdf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:27.980577Z",
     "iopub.status.busy": "2025-01-21T14:04:27.980042Z",
     "iopub.status.idle": "2025-01-21T14:04:27.987008Z",
     "shell.execute_reply": "2025-01-21T14:04:27.986121Z"
    },
    "papermill": {
     "duration": 0.013911,
     "end_time": "2025-01-21T14:04:27.988655",
     "exception": false,
     "start_time": "2025-01-21T14:04:27.974744",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 3])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tensor1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2fddaf4e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:27.998147Z",
     "iopub.status.busy": "2025-01-21T14:04:27.997840Z",
     "iopub.status.idle": "2025-01-21T14:04:28.004398Z",
     "shell.execute_reply": "2025-01-21T14:04:28.002745Z"
    },
    "papermill": {
     "duration": 0.012857,
     "end_time": "2025-01-21T14:04:28.005898",
     "exception": false,
     "start_time": "2025-01-21T14:04:27.993041",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tensor1.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "bdfe9412",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.015499Z",
     "iopub.status.busy": "2025-01-21T14:04:28.015074Z",
     "iopub.status.idle": "2025-01-21T14:04:28.021557Z",
     "shell.execute_reply": "2025-01-21T14:04:28.020412Z"
    },
    "papermill": {
     "duration": 0.013343,
     "end_time": "2025-01-21T14:04:28.023447",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.010104",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "device(type='cpu')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tensor.device"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85601e55",
   "metadata": {
    "papermill": {
     "duration": 0.004261,
     "end_time": "2025-01-21T14:04:28.032407",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.028146",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Tensor operations**\n",
    ">* Tensors support various operations similar to numpy arrays\n",
    ">* tensor operations are compatible only when **r1*c1 ==r2 * c2**\n",
    ">* 1.Addition/Substraction\n",
    ">  a+b , a-b\n",
    ">* 2.element wise multiplication\n",
    ">  a*b\n",
    ">* 3. Transposition\n",
    ">\n",
    ">* 4. Matrix multiplication\n",
    ">   \n",
    ">* 5. Concatination\n",
    "\n",
    "> most numpy array operations can be performed on pytorch tensors "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c2ea6804",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.042411Z",
     "iopub.status.busy": "2025-01-21T14:04:28.041964Z",
     "iopub.status.idle": "2025-01-21T14:04:28.047219Z",
     "shell.execute_reply": "2025-01-21T14:04:28.046188Z"
    },
    "papermill": {
     "duration": 0.012229,
     "end_time": "2025-01-21T14:04:28.048857",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.036628",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "a = torch.tensor([[1,2],[3,4]])\n",
    "b = torch.tensor([[5,6],[7,8]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2b1db91f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.058930Z",
     "iopub.status.busy": "2025-01-21T14:04:28.058538Z",
     "iopub.status.idle": "2025-01-21T14:04:28.100918Z",
     "shell.execute_reply": "2025-01-21T14:04:28.099663Z"
    },
    "papermill": {
     "duration": 0.049605,
     "end_time": "2025-01-21T14:04:28.103003",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.053398",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 5, 12],\n",
       "        [21, 32]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a*b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fe133930",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.113269Z",
     "iopub.status.busy": "2025-01-21T14:04:28.112806Z",
     "iopub.status.idle": "2025-01-21T14:04:28.124142Z",
     "shell.execute_reply": "2025-01-21T14:04:28.122809Z"
    },
    "papermill": {
     "duration": 0.018465,
     "end_time": "2025-01-21T14:04:28.126048",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.107583",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 6,  8],\n",
       "        [10, 12]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a+b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "04ae5205",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.136720Z",
     "iopub.status.busy": "2025-01-21T14:04:28.136379Z",
     "iopub.status.idle": "2025-01-21T14:04:28.156267Z",
     "shell.execute_reply": "2025-01-21T14:04:28.154780Z"
    },
    "papermill": {
     "duration": 0.027396,
     "end_time": "2025-01-21T14:04:28.158325",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.130929",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1, 3],\n",
      "        [2, 4]])\n"
     ]
    }
   ],
   "source": [
    "c = a.T #transposition \n",
    "print(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "848665d3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.169186Z",
     "iopub.status.busy": "2025-01-21T14:04:28.168854Z",
     "iopub.status.idle": "2025-01-21T14:04:28.174669Z",
     "shell.execute_reply": "2025-01-21T14:04:28.173376Z"
    },
    "papermill": {
     "duration": 0.013369,
     "end_time": "2025-01-21T14:04:28.176541",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.163172",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[5, 7],\n",
      "        [6, 8]])\n"
     ]
    }
   ],
   "source": [
    "d = b.T\n",
    "print(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e89f052f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.186820Z",
     "iopub.status.busy": "2025-01-21T14:04:28.186498Z",
     "iopub.status.idle": "2025-01-21T14:04:28.201673Z",
     "shell.execute_reply": "2025-01-21T14:04:28.200568Z"
    },
    "papermill": {
     "duration": 0.022578,
     "end_time": "2025-01-21T14:04:28.203764",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.181186",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "result = torch.cat((a,b),dim = 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8551f08f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.214456Z",
     "iopub.status.busy": "2025-01-21T14:04:28.214061Z",
     "iopub.status.idle": "2025-01-21T14:04:28.220420Z",
     "shell.execute_reply": "2025-01-21T14:04:28.219084Z"
    },
    "papermill": {
     "duration": 0.013627,
     "end_time": "2025-01-21T14:04:28.222215",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.208588",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1, 2, 5, 6],\n",
      "        [3, 4, 7, 8]])\n"
     ]
    }
   ],
   "source": [
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0e4494fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.232922Z",
     "iopub.status.busy": "2025-01-21T14:04:28.232540Z",
     "iopub.status.idle": "2025-01-21T14:04:28.238547Z",
     "shell.execute_reply": "2025-01-21T14:04:28.237446Z"
    },
    "papermill": {
     "duration": 0.013269,
     "end_time": "2025-01-21T14:04:28.240248",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.226979",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 4])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c8498b64",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.251030Z",
     "iopub.status.busy": "2025-01-21T14:04:28.250654Z",
     "iopub.status.idle": "2025-01-21T14:04:28.255340Z",
     "shell.execute_reply": "2025-01-21T14:04:28.254197Z"
    },
    "papermill": {
     "duration": 0.011982,
     "end_time": "2025-01-21T14:04:28.257041",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.245059",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "result1 = torch.cat((c,d))#dim = 1) rows badhegi\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "08cc7cc7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-21T14:04:28.267957Z",
     "iopub.status.busy": "2025-01-21T14:04:28.267565Z",
     "iopub.status.idle": "2025-01-21T14:04:28.273674Z",
     "shell.execute_reply": "2025-01-21T14:04:28.272522Z"
    },
    "papermill": {
     "duration": 0.013401,
     "end_time": "2025-01-21T14:04:28.275450",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.262049",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1, 3],\n",
      "        [2, 4],\n",
      "        [5, 7],\n",
      "        [6, 8]])\n"
     ]
    }
   ],
   "source": [
    "print(result1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56b375d9",
   "metadata": {
    "papermill": {
     "duration": 0.00461,
     "end_time": "2025-01-21T14:04:28.284983",
     "exception": false,
     "start_time": "2025-01-21T14:04:28.280373",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [],
   "dockerImageVersionId": 30839,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 8.443951,
   "end_time": "2025-01-21T14:04:29.713987",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-01-21T14:04:21.270036",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
