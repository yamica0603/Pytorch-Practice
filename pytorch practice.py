{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d8a991e1",
   "metadata": {
    "papermill": {
     "duration": 0.006941,
     "end_time": "2025-01-25T17:58:03.471038",
     "exception": false,
     "start_time": "2025-01-25T17:58:03.464097",
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
   "id": "171f7aad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:03.482922Z",
     "iopub.status.busy": "2025-01-25T17:58:03.482512Z",
     "iopub.status.idle": "2025-01-25T17:58:07.530166Z",
     "shell.execute_reply": "2025-01-25T17:58:07.529016Z"
    },
    "papermill": {
     "duration": 4.055699,
     "end_time": "2025-01-25T17:58:07.532127",
     "exception": false,
     "start_time": "2025-01-25T17:58:03.476428",
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
    "tensor1 = torch.tensor(list1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8afbe4f",
   "metadata": {
    "papermill": {
     "duration": 0.005377,
     "end_time": "2025-01-25T17:58:07.542917",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.537540",
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
   "id": "bd7b8d94",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.554872Z",
     "iopub.status.busy": "2025-01-25T17:58:07.554291Z",
     "iopub.status.idle": "2025-01-25T17:58:07.561598Z",
     "shell.execute_reply": "2025-01-25T17:58:07.560537Z"
    },
    "papermill": {
     "duration": 0.01518,
     "end_time": "2025-01-25T17:58:07.563339",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.548159",
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
   "id": "8b67b371",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.575539Z",
     "iopub.status.busy": "2025-01-25T17:58:07.575209Z",
     "iopub.status.idle": "2025-01-25T17:58:07.580597Z",
     "shell.execute_reply": "2025-01-25T17:58:07.579632Z"
    },
    "papermill": {
     "duration": 0.013306,
     "end_time": "2025-01-25T17:58:07.582299",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.568993",
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
   "id": "8a44af29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.595004Z",
     "iopub.status.busy": "2025-01-25T17:58:07.594590Z",
     "iopub.status.idle": "2025-01-25T17:58:07.600173Z",
     "shell.execute_reply": "2025-01-25T17:58:07.599199Z"
    },
    "papermill": {
     "duration": 0.013907,
     "end_time": "2025-01-25T17:58:07.601896",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.587989",
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
   "id": "bf1b65f7",
   "metadata": {
    "papermill": {
     "duration": 0.005365,
     "end_time": "2025-01-25T17:58:07.612976",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.607611",
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
   "id": "183d56b1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.625166Z",
     "iopub.status.busy": "2025-01-25T17:58:07.624805Z",
     "iopub.status.idle": "2025-01-25T17:58:07.629597Z",
     "shell.execute_reply": "2025-01-25T17:58:07.628475Z"
    },
    "papermill": {
     "duration": 0.013097,
     "end_time": "2025-01-25T17:58:07.631459",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.618362",
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
   "id": "36d09c7c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.643878Z",
     "iopub.status.busy": "2025-01-25T17:58:07.643462Z",
     "iopub.status.idle": "2025-01-25T17:58:07.687726Z",
     "shell.execute_reply": "2025-01-25T17:58:07.686810Z"
    },
    "papermill": {
     "duration": 0.052074,
     "end_time": "2025-01-25T17:58:07.689299",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.637225",
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
   "id": "2193fa48",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.701790Z",
     "iopub.status.busy": "2025-01-25T17:58:07.701421Z",
     "iopub.status.idle": "2025-01-25T17:58:07.711636Z",
     "shell.execute_reply": "2025-01-25T17:58:07.710577Z"
    },
    "papermill": {
     "duration": 0.018685,
     "end_time": "2025-01-25T17:58:07.713711",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.695026",
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
   "id": "e455396f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.727722Z",
     "iopub.status.busy": "2025-01-25T17:58:07.727293Z",
     "iopub.status.idle": "2025-01-25T17:58:07.747817Z",
     "shell.execute_reply": "2025-01-25T17:58:07.746570Z"
    },
    "papermill": {
     "duration": 0.029735,
     "end_time": "2025-01-25T17:58:07.749768",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.720033",
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
   "id": "1317db5c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.763009Z",
     "iopub.status.busy": "2025-01-25T17:58:07.762588Z",
     "iopub.status.idle": "2025-01-25T17:58:07.768594Z",
     "shell.execute_reply": "2025-01-25T17:58:07.767484Z"
    },
    "papermill": {
     "duration": 0.014369,
     "end_time": "2025-01-25T17:58:07.770172",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.755803",
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
   "id": "66fb048a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.783142Z",
     "iopub.status.busy": "2025-01-25T17:58:07.782792Z",
     "iopub.status.idle": "2025-01-25T17:58:07.800536Z",
     "shell.execute_reply": "2025-01-25T17:58:07.799518Z"
    },
    "papermill": {
     "duration": 0.026236,
     "end_time": "2025-01-25T17:58:07.802386",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.776150",
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
   "id": "8db50b28",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.815491Z",
     "iopub.status.busy": "2025-01-25T17:58:07.815143Z",
     "iopub.status.idle": "2025-01-25T17:58:07.821045Z",
     "shell.execute_reply": "2025-01-25T17:58:07.819933Z"
    },
    "papermill": {
     "duration": 0.014391,
     "end_time": "2025-01-25T17:58:07.822891",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.808500",
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
   "id": "83967003",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.836622Z",
     "iopub.status.busy": "2025-01-25T17:58:07.836294Z",
     "iopub.status.idle": "2025-01-25T17:58:07.841562Z",
     "shell.execute_reply": "2025-01-25T17:58:07.840776Z"
    },
    "papermill": {
     "duration": 0.013645,
     "end_time": "2025-01-25T17:58:07.842993",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.829348",
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
   "id": "ad454b7d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.856448Z",
     "iopub.status.busy": "2025-01-25T17:58:07.856101Z",
     "iopub.status.idle": "2025-01-25T17:58:07.860580Z",
     "shell.execute_reply": "2025-01-25T17:58:07.859490Z"
    },
    "papermill": {
     "duration": 0.013123,
     "end_time": "2025-01-25T17:58:07.862293",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.849170",
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
   "id": "faabf7a5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.876120Z",
     "iopub.status.busy": "2025-01-25T17:58:07.875710Z",
     "iopub.status.idle": "2025-01-25T17:58:07.881530Z",
     "shell.execute_reply": "2025-01-25T17:58:07.880654Z"
    },
    "papermill": {
     "duration": 0.014505,
     "end_time": "2025-01-25T17:58:07.883037",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.868532",
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
   "cell_type": "markdown",
   "id": "245b3f4f",
   "metadata": {
    "papermill": {
     "duration": 0.006188,
     "end_time": "2025-01-25T17:58:07.895510",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.889322",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# CREATING OUR FIRST NEURAL NETWORK \n",
    "* exploring how neural networks take input perform computations and produce outputs.\n",
    "* we will use **torch.nn** package to **create** our networks. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "50e9a0eb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.909055Z",
     "iopub.status.busy": "2025-01-25T17:58:07.908608Z",
     "iopub.status.idle": "2025-01-25T17:58:07.912932Z",
     "shell.execute_reply": "2025-01-25T17:58:07.911819Z"
    },
    "papermill": {
     "duration": 0.012997,
     "end_time": "2025-01-25T17:58:07.914714",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.901717",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# creating a basic NN with only input and output layer \n",
    "import torch.nn as nn"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "282439a5",
   "metadata": {
    "papermill": {
     "duration": 0.00589,
     "end_time": "2025-01-25T17:58:07.926919",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.921029",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "\n",
    "* first layer will be an **input layer**.\n",
    "* Create input tensor with 3 features(neurons) using torch.tensor([[1,2,3],[4,5,6]]).\n",
    "* next step is to apply linear layer [a linear layer takes an input, applies a linear function and returns output].\n",
    "* linear layer takes in\n",
    "* 1. no of features in input layer **(in_features)** ensures linear layer recieves the input tensor and\n",
    "  2. 2. no. of output freatures **(out_features)** using **nn.linear function** .\n",
    "*  lastly we pass the input tensor to linear layer to generate an output .\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fb6c46e",
   "metadata": {
    "papermill": {
     "duration": 0.005676,
     "end_time": "2025-01-25T17:58:07.938579",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.932903",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#Getting to know the linear layer\n",
    "\n",
    "* each layer has a **.weight** and **.bias** property\n",
    "* what operation does nn.linear perform?\n",
    "*  when the input layer passes through the linear layer the operation performed is a matrix multiplication between the input_tensors and the weight followed by adding the bias.\n",
    "*  y = w*x + b - nn.linear function takes care of this \n",
    "\n",
    "* initially when we call nn.linear the weights and biases are initialized randomly, so they are yet not useful\n",
    "* networks with only linear layers are called **fully connected**.\n",
    "* ( 1 col * n rows - input layer & 1 col * n rows for linear output layer )\n",
    "* stacking multiple layers sequentially using **nn.sequential** function\n",
    "*  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fbc4abce",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:07.952014Z",
     "iopub.status.busy": "2025-01-25T17:58:07.951597Z",
     "iopub.status.idle": "2025-01-25T17:58:08.003538Z",
     "shell.execute_reply": "2025-01-25T17:58:08.002210Z"
    },
    "papermill": {
     "duration": 0.060774,
     "end_time": "2025-01-25T17:58:08.005505",
     "exception": false,
     "start_time": "2025-01-25T17:58:07.944731",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[-0.3975, -0.1985]], grad_fn=<AddmmBackward0>)\n"
     ]
    }
   ],
   "source": [
    "#linear model \n",
    "\n",
    "input_tensor = torch.tensor(\n",
    "    [[0.3471,0.4547,-0.2356]]\n",
    ")\n",
    "\n",
    "# defne our linear layer \n",
    "\n",
    "linear_layer = nn.Linear(in_features = 3, out_features = 2)\n",
    "\n",
    "#pass input though linear layer \n",
    "\n",
    "output = linear_layer(input_tensor)\n",
    "print (output)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f71a0941",
   "metadata": {
    "papermill": {
     "duration": 0.006499,
     "end_time": "2025-01-25T17:58:08.020062",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.013563",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Discovering activation functions "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0d920f3",
   "metadata": {
    "papermill": {
     "duration": 0.006346,
     "end_time": "2025-01-25T17:58:08.033210",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.026864",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "we will now add non linearity to our models using **activation functions** \n",
    "> * Non linearity helps the model to learn more complex relationships.\n",
    "> * preactivation output is the input for activation function\n",
    "> * Sigmaoid activaton function - used for binary classification problems.\n",
    "> * e.g. - animal --> mamal or not ?\n",
    "> * nn.sigmoid() is used in last layer of the neural network\n",
    "> * sigmoid as a last step in nettwork of linear layers is equivalent to traditional logistic regression.\n",
    "> * nn.Softmax() is used for **multiclass**  classification.\n",
    "> * "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6d4bf277",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:08.047147Z",
     "iopub.status.busy": "2025-01-25T17:58:08.046766Z",
     "iopub.status.idle": "2025-01-25T17:58:08.054034Z",
     "shell.execute_reply": "2025-01-25T17:58:08.052903Z"
    },
    "papermill": {
     "duration": 0.015992,
     "end_time": "2025-01-25T17:58:08.055655",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.039663",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[0.7685, 0.9608, 0.9866]])\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "input_tensor1 = torch.tensor([[1.2,3.2,4.3]])\n",
    "\n",
    "#applying sigmoid function \n",
    "\n",
    "probability = nn.Sigmoid()\n",
    "output_tensor1 = probability(input_tensor1)\n",
    "\n",
    "print(output_tensor1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "493f5899",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:08.069942Z",
     "iopub.status.busy": "2025-01-25T17:58:08.069546Z",
     "iopub.status.idle": "2025-01-25T17:58:08.081047Z",
     "shell.execute_reply": "2025-01-25T17:58:08.079616Z"
    },
    "papermill": {
     "duration": 0.020849,
     "end_time": "2025-01-25T17:58:08.083163",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.062314",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[0.6095]], grad_fn=<SigmoidBackward0>)\n"
     ]
    }
   ],
   "source": [
    "#adding sigmoid function to linear layers \n",
    "\n",
    "import torch \n",
    "import torch.nn as nn\n",
    "\n",
    "input_tensor = torch.tensor([[4.3,6.1,2.3]])\n",
    "\n",
    "# creating a linear layer neural network \n",
    "\n",
    "Linear_layer = nn.Sequential(\n",
    "    nn.Linear (3,3),\n",
    "    nn.Linear(3,2),\n",
    "    nn.Linear(2,1),\n",
    "    nn.Sigmoid()\n",
    ")\n",
    "\n",
    "output = Linear_layer(input_tensor)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "02ee632b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-25T17:58:08.096980Z",
     "iopub.status.busy": "2025-01-25T17:58:08.096577Z",
     "iopub.status.idle": "2025-01-25T17:58:08.109902Z",
     "shell.execute_reply": "2025-01-25T17:58:08.108556Z"
    },
    "papermill": {
     "duration": 0.022241,
     "end_time": "2025-01-25T17:58:08.111759",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.089518",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[0.1392, 0.8420, 0.0188]])\n"
     ]
    }
   ],
   "source": [
    "import torch \n",
    "import torch.nn as nn\n",
    "\n",
    "#create an input tensor \n",
    "input_tensor = torch.tensor([[4.3,6.1,2.3]])\n",
    "\n",
    "#apply softmax function along that last dimension\n",
    "probabilities = nn.Softmax(dim = 1)\n",
    "output_tensor = probabilities(input_tensor)\n",
    "\n",
    "print(output_tensor)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08a3e296",
   "metadata": {
    "papermill": {
     "duration": 0.006743,
     "end_time": "2025-01-25T17:58:08.125021",
     "exception": false,
     "start_time": "2025-01-25T17:58:08.118278",
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
   "duration": 9.106957,
   "end_time": "2025-01-25T17:58:09.655321",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-01-25T17:58:00.548364",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
