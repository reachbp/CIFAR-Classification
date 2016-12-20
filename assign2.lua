require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'

--[[

This file shows the modified example from the paper "Torchnet: An Open-Source Platform
for (Deep) Learning Research".

Revisions by Rob Fergus (fergus@cs.nyu.edu) and Christian Puhrsch (cpuhrsch@fb.com)
Version 1.0 (10/14/16)

--]]

local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 100, 'batchsize')
cmd:option('-mnist', false, 'use mnist')
cmd:option('-cifar', false, 'use cifar')
cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
--local dbg   = require 'debugger'
-- to set breakpoint put just put: dbg() at desired line

local base_data_path = '/work/bt978/cifardata/'

--- Question 2. Display images
--[[
local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
-- Printing the first 100 images
print("Printing the first 100 Images of MNIST")
image.display{image = train.data[{{1,100}}], legend = 'First 100 MNIST', scaleeach =true}
print("Printing the first 100 Images of CIFAR")
local train_cifar = torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii')
print({train_cifar})
local first100_cifar = train_cifar.data:permute(2,1):reshape(10000,3,32,32)[{{1,100},{},{},{}}]
print({first100_cifar})
image.display{image = first100_cifar, legend = 'First 100 CIFAR', scaleeach =true}
--]]
-- Dataprep for MNIST
if config.mnist == true and false then
    if not paths.filep(base_data_path .. 'train_small_28x28.t7') then
        print("loading MNIST data")
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local train_small = {}
        train_small.data   = train.data[{{1, 50000}, {}, {}, {}}]
        train_small.labels = train.labels[{{1, 50000}}]
        torch.save(base_data_path .. 'train_small_28x28.t7', train_small, 'ascii')
    end
    if not paths.filep(base_data_path .. 'valid.t7') then
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local valid = {}
        valid.data   = train.data[{{50001, 60000}, {}, {}, {}}]
        valid.labels = train.labels[{{50001, 60000}}]
        torch.save(base_data_path .. 'valid_28x28.t7', valid, 'ascii')
    end
end

------------------------------------------------------------------------
-- Build the dataloader

-- getDatasets returns a dataset that performs some minor transformation on
-- the input and the target (TransformDataset), shuffles the order of the
-- samples without replacement (ShuffleDataset) and merges them into
-- batches (BatchDataset).
local function getMnistIterator(datasets, useFull)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
     local list
      if(useFull) then
           list = torch.range(1, dataset.data:size(1)):totable()
      else
           list = torch.range(1, 1000):totable()
      end
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[idx],
                                target = dataset.labels[idx]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		       return {
			  input  = x.input:view(-1):double(), torch.zeros(240),
                    transform = function(x)
		       return {
			  input  = torch.cat(x.input:view(-1):double(), torch.zeros(240)),
			  target = torch.LongTensor{x.target }
                        }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end

local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
        local list = torch.range(1, dataset.data:size(1)):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[{{}, idx}],
                                target = dataset.labels[{{}, idx}]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		       return {
			  input  = x.input:double():reshape(3,32,32),
			  target = x.target:long():add(1),
		       }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end

------------------------------------------------------------------------
-- Make the model and the criterion

local nout = 10 --same for both CIFAR and MNIST
local nin

if config.mnist == true then nin = 1024 end
if config.cifar == true then nin = 3072 end

local network = nn.Linear(nin, nout)
local criterion = nn.CrossEntropyCriterion()

 --- 3.b Adding a Tanh non-linearity
local linear1 = nn.Linear(nin, 1000)
local tanh1 = nn.Tanh()
local linear2 = nn.Linear(1000, nout)
local network = nn.Sequential()
network:add(linear1)
network:add(tanh1)
network:add(linear2)


--- Qn 5.
local net1 = nn.Sequential()
net1:add(nn.SpatialConvolution(3, 16, 5, 5))
net1:add(nn.Tanh())
net1:add(nn.SpatialMaxPooling(2,2,2, 2))
net1:add(nn.SpatialConvolution(16, 128, 5, 5))
net1:add(nn.Tanh())
net1:add(nn.SpatialMaxPooling(2,2,2, 2))
net1:add(nn.View(128*5*5))
net1:add(nn.Linear(128*5*5, 64))
net1:add(nn.Tanh())
net1:add(nn.Linear(64,10))


if config.cifar == true then network = net1 end
------------------------------------------------------------------------
-- Prepare torchnet environment for training and testing
--network = net1
local trainiterator
local validiterator
local testiterator
if config.mnist == true then
    local datasets
    local genData = torch.load(base_data_path ..'genData.t7') 
    print("Striucture of generatred data", genData)
    datasets = {torch.load(base_data_path .. 'old_mnist/train_28x28.t7', 'ascii')}
    print("Structure of actual mnsit data", datasets[1])
    trainiterator = getMnistIterator(datasets, false)
    datasets = {torch.load(base_data_path .. 'old_mnist/train_28x28.t7', 'ascii')}
    --datasets = {torch.load(base_data_path .. 'valid_28x28.t7', 'ascii')}
 print("Structure of actual training data", datasets[1])  
  validiterator = getMnistIterator(datasets, false)
    datasets = {torch.load(base_data_path .. 'test_32x32_new.t7', 'ascii')}
   print("Structure of actual  test data", datasets[1])
    --datasets = {torch.load(base_data_path .. 'test_28x28.t7', 'ascii')}
    testiterator  = getMnistIterator(datasets, true)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
    trainiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
    validiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
    testiterator  = getCifarIterator(datasets)
end

local lr = config.lr
local epochs = config.epochs

print("Started training!")

for epoch = 1, epochs do
    local timer = torch.Timer()
    local loss = 0
    local errors = 0
    local count = 0
    for d in trainiterator() do
        network:forward(d.input)
        criterion:forward(network.output, d.target)
        network:zeroGradParameters()
        criterion:backward(network.output, d.target)
        network:backward(d.input, criterion.gradInput)
        network:updateParameters(lr)
        loss = loss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        errors = errors + (pred:size(1) - pred:eq(d.target):sum())
    end
    loss = loss / count
    local validloss = 0
    local validerrors = 0
    count = 0
    for d in validiterator() do
        network:forward(d.input)
        criterion:forward(network.output, d.target)
        validloss = validloss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        validerrors = validerrors + (pred:size(1) - pred:eq(d.target):sum())
    end
    validloss = validloss / count
    print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f - valid | validloss: %2.4f | validerror: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, errors, validloss, validerrors, timer:time().real
    ))
end
--- Printing out the output of Tanh Layer
---print(network:get(2).output)

---Add code to plot out the network weights as images
----(one for each out- put, of size 28 by 28) after the last epoch.
----Grab a screenshot of the figure and include it in your report.
--- Qn 5: Print out the parameters for each Layer
--[[
for i  = 1, 10 do
  local params = network:get(i):parameters()
  if params ~= nil then
    local weights = params[1]
    local bias = params[2]
    print("Layer ", i, " Weights:", weights:nElement(), "Biases:", bias:nElement())
  end
end
--]]
print("Printing out network weights")
--image.display(network:get(1).weight)
local testerrors = 0
count = 0
for d in testiterator() do
    
    network:forward(d.input)
    print("Network output", network.output[1], "Target", d.target[1])
    criterion:forward(network.output, d.target)
    local _, pred = network.output:max(2)
    testerrors = testerrors + (pred:size(1) - pred:eq(d.target):sum())
    count = count + 1
end
print(string.format('| test | error: %2.4f', testerrors))
---image.display(network.weight:reshape(10,28,28))
