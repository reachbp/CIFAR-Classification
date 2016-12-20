require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'env'
require 'trepl'
require 'cunn'


local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 100, 'batchsize')
cmd:option('-mnist', false, 'use mnist')
cmd:option('-cifar', false, 'use cifar')
cmd:option('-epochs', 100 , 'epochs')
cmd:option('-npc', 100 , 'numperclass')
cmd:option('-output', 'output1', 'output file')
cmd:option('-type','cuda','type')
cmd:option('-model','vgg','model_type')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
--local dbg   = require 'debugger'
-- to set breakpoint put just put: dbg() at desired line

local base_data_path = '/work/bt978/cifardata/'
local npc = config.npc
local opt = config

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
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
			  input  = x.input:view(-1):double(),
			  target = torch.LongTensor{x.target + 1}
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
    --inp.data = torch.reshape(inp.data, inp.size(), 3072):t()
	--inp.labels = torch.reshape(inp.labels, 1, inp.size())
    for _, dataset in pairs(datasets) do
--	inp = dataset
  --    inp.data = torch.reshape(inp.data, inp.size(), 3072):t()
--inp.labels = torch.reshape(inp.labels, 1, inp.size())
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

local logs = {
   train_loss_full = {},
   train_loss = {},
   val_loss = {},
   map = {},
   clerr = {},
}
function log(logs)
        local gnuplot = require 'gnuplot'
        gnuplot.pngfigure(paths.concat('outputs' ,config.output .. 'valtrain_' .. #logs.train_loss ..'.png'))
        gnuplot.plot({'train loss', torch.range(1, #logs.train_loss),torch.Tensor(logs.train_loss)}, {'val loss',torch.Tensor(logs.val_loss)})
        gnuplot.title('loss per epoch' .. config.output)
   gnuplot.plotflush()
end


local nout = 10 --same for both CIFAR and MNIST
local nin

if config.mnist == true then nin = 784 end
if config.cifar == true then nin = 3072 end

local network = nn.Linear(nin, nout)
local criterion = nn.CrossEntropyCriterion():cuda()

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
--model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
--model:add(cast(dofile('vgg2.lua')))
--model:get(2).updateGradInput = function(input) return end

--print(model)


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
net1:cuda()


confusion = optim.ConfusionMatrix(10)



if config.cifar == true then network = net1 end
------------------------------------------------------------------------
-- Prepare torchnet environment for training and testing

local trainiterator
local validiterator
local testiterator
if config.mnist == true then
    local datasets
    datasets = {torch.load(base_data_path .. 'train_small_28x28.t7', 'ascii')}
    trainiterator = getMnistIterator(datasets, false)
    datasets = {torch.load(base_data_path .. 'valid_28x28.t7', 'ascii')}
    validiterator = getMnistIterator(datasets, false)
    datasets = {torch.load(base_data_path .. 'test_28x28.t7', 'ascii')}
    testiterator  = getMnistIterator(datasets, true)
end
if config.cifar == true then
    local datasets
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/batch1_gd' .. npc .. '.t7'),
	torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
	torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
	torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
    print(datasets[1].data:size())
    trainiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
    validiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
    testiterator  = getCifarIterator(datasets)
end



local lr = config.lr
local epochs = config.epochs

print("Started training!")
local train_out = assert(io.open("outputs/" .. config.output ..  "train.csv", "w"))
local val_out = assert(io.open("outputs/".. config.output .. "val.csv", "w"))

for epoch = 1, epochs do
    local timer = torch.Timer()
    local loss = 0
    local errors = 0
    local count = 0
    for d in trainiterator() do
        local output = network:forward(d.input:cuda())
        local f = criterion:forward(output, d.target:cuda())
        network:zeroGradParameters()
        local df_do = criterion:backward(output, d.target:cuda())
        network:backward(d.input:cuda(), df_do)
	confusion:batchAdd(output:double(), d.target:double())
        network:updateParameters(lr)
        loss = loss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        errors = errors + (pred:size(1) - pred:eq(d.target:cuda()):sum())
    end
    loss = loss / count
    train_out:write(loss,  "\n")
    logs.train_loss[#logs.train_loss + 1] = loss    
    local validloss = 0
    local validerrors = 0
    confusion:updateValids()
    local train_acc = confusion.totalValid*100
    local val_acc = 0
    confusion:zero()
    count = 0
    for d in validiterator() do
        local output = network:forward(d.input:cuda())
        criterion:forward(network.output, d.target:cuda())
        confusion:batchAdd(output, d.target:double())
	validloss = validloss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        validerrors = validerrors + (pred:size(1) - pred:eq(d.target:cuda()):sum())
    end
    confusion:updateValids()
    val_acc = confusion.totalValid*100
    confusion:zero()
    validloss = validloss / count
    val_out:write(validloss, "\n")
    logs.val_loss[#logs.val_loss+ 1] = validloss   
     print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | acc: %2.4f - valid | validloss: %2.4f | validacc: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, train_acc, validloss, val_acc, timer:time().real
    ))
    confusion:zero()
end
--- Printing out the output of Tanh Layer
---print(network:get(2).output)
train_out:close()
val_out:close()
-- logging the graph
print(logs.train_loss)
log(logs)


print("Printing out network weights")
--image.display(network:get(1).weight)
local testerrors = 0
count = 0
for d in testiterator() do
    network:forward(d.input:cuda())
    ----print("Network output", network.output[1], "Target", d.target[1])
    criterion:forward(network.output, d.target:cuda())
    local _, pred = network.output:max(2)
    testerrors = testerrors + (pred:size(1) - pred:eq(d.target:cuda()):sum())
    count = count + 1
end
--print(string.format('| test | error: %2.4f', testerrors))
---image.display(network.weight:reshape(10,28,28))
