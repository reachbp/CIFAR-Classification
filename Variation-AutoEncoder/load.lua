require 'hdf5'
require 'torch'

function load(continuous)
    if continuous then
        return loadfreyfaces()
    else
        return loadmnist()
    end
end

function loadmnist()
    -- This loads an hdf5 version of the MNIST dataset used here: http://deeplearning.net/tutorial/gettingstarted.html
    -- Direct link: http://deeplearning.net/data/mnist/mnist.pkl.gz
   
   --[[
    local f = hdf5.open('datasets/mnist.hdf5', 'r')
    data = {}
    data.train = f:read('x_train'):all():double()
    data.test = f:read('x_test'):all():double()
    print(data.train[1])
    --print(data.test:size())
    --data.test_labels = f:read('train')
    f:close()
--]]
    
    train_datasets = torch.load('datasets/mnist.t7/train_32x32.t7', 'ascii')
    test_datasets = torch.load('datasets/mnist.t7/test_32x32.t7', 'ascii')
    
    print(train_datasets[1])
    data = {}
    train = {}
    data.train = train_datasets.data:double()
    data.train = data.train:permute(1, 3, 4, 2)
     --print(data.train:size())
    data.train = torch.reshape(data.train,60000,   1024):double():div(255)
    data.test = {}
    data.test.labels = test_datasets.labels:double()
    data.test.data = test_datasets.data:double()
    data.test.data = data.test.data:permute(1, 3, 4, 2)
    data.test.data = torch.reshape(data.test.data, 10000, 1024):double():div(255)
    print("Size of training dataset loaded ", data.train:size())
    print("Size of test dataset loaded ", data.test.data:size())
    
    return data
end

function loadfreyfaces()
    require 'hdf5'
    local f = hdf5.open('datasets/freyfaces.hdf5', 'r')
    local data = {}
    data.train = f:read('train'):all():double()
    data.test = f:read('test'):all():double()
    f:close()

    return data
end

loadmnist()
