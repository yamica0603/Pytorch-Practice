{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":30839,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"markdown","source":"# Pytorch\n * Tensors is the fundamental data structure in pytorch.\n * it is an array which performs mathematical operations and is the building block of neural networks\n * can be created form python **lists** using **tensor.torch(list_name)** or from **numpy** array using toch.fro_numpy(**list_name**)\n * similar to numpyarrays tensors are **multidimensional** representations of their elements.\n","metadata":{}},{"cell_type":"code","source":"import torch\n\nlist =  [[1,2,3],[4,5,6]]\ntensor = torch.tensor(list) #creation of tensors using lists \n\nlist1 = [[7,8,9],[1,2,3]]\ntensor1 = torch.tensor(list1)","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:54.260145Z","iopub.execute_input":"2025-01-24T17:47:54.260593Z","iopub.status.idle":"2025-01-24T17:47:58.576764Z","shell.execute_reply.started":"2025-01-24T17:47:54.260545Z","shell.execute_reply":"2025-01-24T17:47:58.575671Z"}},"outputs":[],"execution_count":2},{"cell_type":"markdown","source":"#Tensor attributes\n>* Tensor shape - var_name.shape\n>* Tensor datatype - var_name.dtype\n>* check which devce the tensor is loaded on (cpu/gpu) - var_name.device\n","metadata":{}},{"cell_type":"code","source":"tensor1.shape","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.578038Z","iopub.execute_input":"2025-01-24T17:47:58.578434Z","iopub.status.idle":"2025-01-24T17:47:58.585088Z","shell.execute_reply.started":"2025-01-24T17:47:58.578408Z","shell.execute_reply":"2025-01-24T17:47:58.583933Z"}},"outputs":[{"execution_count":3,"output_type":"execute_result","data":{"text/plain":"torch.Size([2, 3])"},"metadata":{}}],"execution_count":3},{"cell_type":"code","source":"tensor1.dtype","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.587422Z","iopub.execute_input":"2025-01-24T17:47:58.587797Z","iopub.status.idle":"2025-01-24T17:47:58.612159Z","shell.execute_reply.started":"2025-01-24T17:47:58.587771Z","shell.execute_reply":"2025-01-24T17:47:58.611191Z"}},"outputs":[{"execution_count":4,"output_type":"execute_result","data":{"text/plain":"torch.int64"},"metadata":{}}],"execution_count":4},{"cell_type":"code","source":"tensor.device","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.613939Z","iopub.execute_input":"2025-01-24T17:47:58.614331Z","iopub.status.idle":"2025-01-24T17:47:58.632270Z","shell.execute_reply.started":"2025-01-24T17:47:58.614290Z","shell.execute_reply":"2025-01-24T17:47:58.631316Z"}},"outputs":[{"execution_count":5,"output_type":"execute_result","data":{"text/plain":"device(type='cpu')"},"metadata":{}}],"execution_count":5},{"cell_type":"markdown","source":"# **Tensor operations**\n>* Tensors support various operations similar to numpy arrays\n>* tensor operations are compatible only when **r1*c1 ==r2 * c2**\n>* 1.Addition/Substraction\n>  a+b , a-b\n>* 2.element wise multiplication\n>  a*b\n>* 3. Transposition\n>\n>* 4. Matrix multiplication\n>   \n>* 5. Concatination\n\n> most numpy array operations can be performed on pytorch tensors ","metadata":{}},{"cell_type":"code","source":"a = torch.tensor([[1,2],[3,4]])\nb = torch.tensor([[5,6],[7,8]])\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.633260Z","iopub.execute_input":"2025-01-24T17:47:58.633526Z","iopub.status.idle":"2025-01-24T17:47:58.654654Z","shell.execute_reply.started":"2025-01-24T17:47:58.633503Z","shell.execute_reply":"2025-01-24T17:47:58.653345Z"}},"outputs":[],"execution_count":6},{"cell_type":"code","source":"a*b","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.655826Z","iopub.execute_input":"2025-01-24T17:47:58.656218Z","iopub.status.idle":"2025-01-24T17:47:58.718249Z","shell.execute_reply.started":"2025-01-24T17:47:58.656177Z","shell.execute_reply":"2025-01-24T17:47:58.717165Z"}},"outputs":[{"execution_count":7,"output_type":"execute_result","data":{"text/plain":"tensor([[ 5, 12],\n        [21, 32]])"},"metadata":{}}],"execution_count":7},{"cell_type":"code","source":"a+b","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.719389Z","iopub.execute_input":"2025-01-24T17:47:58.719695Z","iopub.status.idle":"2025-01-24T17:47:58.730501Z","shell.execute_reply.started":"2025-01-24T17:47:58.719667Z","shell.execute_reply":"2025-01-24T17:47:58.729468Z"}},"outputs":[{"execution_count":8,"output_type":"execute_result","data":{"text/plain":"tensor([[ 6,  8],\n        [10, 12]])"},"metadata":{}}],"execution_count":8},{"cell_type":"code","source":"c = a.T #transposition \nprint(c)","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.733395Z","iopub.execute_input":"2025-01-24T17:47:58.733702Z","iopub.status.idle":"2025-01-24T17:47:58.762617Z","shell.execute_reply.started":"2025-01-24T17:47:58.733676Z","shell.execute_reply":"2025-01-24T17:47:58.761684Z"}},"outputs":[{"name":"stdout","text":"tensor([[1, 3],\n        [2, 4]])\n","output_type":"stream"}],"execution_count":9},{"cell_type":"code","source":"d = b.T\nprint(d)","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.763983Z","iopub.execute_input":"2025-01-24T17:47:58.764247Z","iopub.status.idle":"2025-01-24T17:47:58.769979Z","shell.execute_reply.started":"2025-01-24T17:47:58.764223Z","shell.execute_reply":"2025-01-24T17:47:58.768981Z"}},"outputs":[{"name":"stdout","text":"tensor([[5, 7],\n        [6, 8]])\n","output_type":"stream"}],"execution_count":10},{"cell_type":"code","source":"result = torch.cat((a,b),dim = 1)\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.770957Z","iopub.execute_input":"2025-01-24T17:47:58.771284Z","iopub.status.idle":"2025-01-24T17:47:58.802247Z","shell.execute_reply.started":"2025-01-24T17:47:58.771257Z","shell.execute_reply":"2025-01-24T17:47:58.801205Z"}},"outputs":[],"execution_count":11},{"cell_type":"code","source":"print(result)","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.803316Z","iopub.execute_input":"2025-01-24T17:47:58.803633Z","iopub.status.idle":"2025-01-24T17:47:58.811045Z","shell.execute_reply.started":"2025-01-24T17:47:58.803605Z","shell.execute_reply":"2025-01-24T17:47:58.809955Z"}},"outputs":[{"name":"stdout","text":"tensor([[1, 2, 5, 6],\n        [3, 4, 7, 8]])\n","output_type":"stream"}],"execution_count":12},{"cell_type":"code","source":"result.shape\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.812229Z","iopub.execute_input":"2025-01-24T17:47:58.812548Z","iopub.status.idle":"2025-01-24T17:47:58.834926Z","shell.execute_reply.started":"2025-01-24T17:47:58.812518Z","shell.execute_reply":"2025-01-24T17:47:58.833835Z"}},"outputs":[{"execution_count":13,"output_type":"execute_result","data":{"text/plain":"torch.Size([2, 4])"},"metadata":{}}],"execution_count":13},{"cell_type":"code","source":"result1 = torch.cat((c,d))#dim = 1) rows badhegi\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.836334Z","iopub.execute_input":"2025-01-24T17:47:58.836659Z","iopub.status.idle":"2025-01-24T17:47:58.851199Z","shell.execute_reply.started":"2025-01-24T17:47:58.836632Z","shell.execute_reply":"2025-01-24T17:47:58.850231Z"}},"outputs":[],"execution_count":14},{"cell_type":"code","source":"print(result1)","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.852305Z","iopub.execute_input":"2025-01-24T17:47:58.852570Z","iopub.status.idle":"2025-01-24T17:47:58.871251Z","shell.execute_reply.started":"2025-01-24T17:47:58.852546Z","shell.execute_reply":"2025-01-24T17:47:58.870190Z"}},"outputs":[{"name":"stdout","text":"tensor([[1, 3],\n        [2, 4],\n        [5, 7],\n        [6, 8]])\n","output_type":"stream"}],"execution_count":15},{"cell_type":"markdown","source":"# CREATING OUR FIRST NEURAL NETWORK \n* exploring how neural networks take input perform computations and produce outputs.\n* we will use **torch.nn** package to **create** our networks. ","metadata":{}},{"cell_type":"code","source":"# creating a basic NN with only input and output layer \nimport torch.nn as nn","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:47:58.872256Z","iopub.execute_input":"2025-01-24T17:47:58.872617Z","iopub.status.idle":"2025-01-24T17:47:58.889070Z","shell.execute_reply.started":"2025-01-24T17:47:58.872588Z","shell.execute_reply":"2025-01-24T17:47:58.887612Z"}},"outputs":[],"execution_count":16},{"cell_type":"markdown","source":"\n* first layer will be an **input layer**.\n* Create input tensor with 3 features(neurons) using torch.tensor([[1,2,3],[4,5,6]]).\n* next step is to apply linear layer [a linear layer takes an input, applies a linear function and returns output].\n* linear layer takes in\n* 1. no of features in input layer **(in_features)** ensures linear layer recieves the input tensor and\n  2. 2. no. of output freatures **(out_features)** using **nn.linear function** .\n*  lastly we pass the input tensor to linear layer to generate an output .\n","metadata":{}},{"cell_type":"code","source":"#linear model \n\ninput_tensor = torch.tensor(\n    [[0.3471,0.4547,-0.2356]]\n)\n\n# defne our linear layer \n\nlinear_layer = nn.Linear(in_features = 3, out_features = 2)\n\n#pass input though linear layer \n\noutput = linear_layer(input_tensor)\nprint (output)\n\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-01-24T17:50:03.492146Z","iopub.execute_input":"2025-01-24T17:50:03.492550Z","iopub.status.idle":"2025-01-24T17:50:03.543655Z","shell.execute_reply.started":"2025-01-24T17:50:03.492520Z","shell.execute_reply":"2025-01-24T17:50:03.542473Z"}},"outputs":[{"name":"stdout","text":"tensor([[0.0982, 0.1710]], grad_fn=<AddmmBackward0>)\n","output_type":"stream"}],"execution_count":20},{"cell_type":"markdown","source":"#Getting to know the linear layer\n\n* each layer has a **.weight** and **.bias** property\n* what operation does nn.linear perform?\n*  when the input layer passes through the linear layer the operation performed is a matrix multiplication between the input_tensors and the weight followed by adding the bias.\n*  y = w*x + b - nn.linear function takes care of this \n\n* initially when we call nn.linear the weights and biases are initialized randomly, so they are yet not useful\n* networks with only linear layers are called **fully connected**.\n* ( 1 col * n rows - input layer & 1 col * n rows for linear output layer )\n* stacking multiple layers sequentially using **nn.sequential** function\n*  \n","metadata":{}},{"cell_type":"code","source":"","metadata":{"trusted":true},"outputs":[],"execution_count":null}]}