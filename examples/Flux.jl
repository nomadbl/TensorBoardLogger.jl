#For a Colab of this example, goto https://colab.research.google.com/drive/1xfUsBn9GEqbRjBF-UX_jnGjHZNtNsMae
using TensorBoardLogger
using Flux, ChainRulesCore
using Logging
using MLDatasets
using Statistics

#create tensorboard logger
logdir = "Fluxlogs"
image_logger = TBLogger(logdir, tb_append; prefix="image")
train_logger = TBLogger(logdir, tb_append; prefix="train") # global_logger() # default used to check events are trying to log
val_logger = TBLogger(logdir, tb_append; prefix="val") # global_logger()

#Load data
traindata, trainlabels = FashionMNIST.traindata();
testdata, testlabels = FashionMNIST.testdata();
trainsize = size(traindata, 3);
testsize = size(testdata, 3);

#Log some images
images = TBImage(traindata[:, :, 1:10], WHN)
with_logger(val_logger) do #log some samples
    @info "fmnist/samples" pics = images log_step_increment=0
end

#Create model
model = Chain(
    x -> reshape(x, :, size(x, 4)),
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax
) |> gpu

loss_fn(pred, y) = Flux.crossentropy(pred, y)

accuracy(pred, y) = mean(Flux.onecold(pred |> cpu) .== Flux.onecold(y |> cpu))

opt = Flux.setup(ADAM(), model)

traindata = reshape(traindata, (28, 28, 1, 60000));
testdata = reshape(testdata, (28, 28, 1, 10000));
trainlabels = Flux.onehotbatch(trainlabels, collect(0:9));
testlabels = Flux.onehotbatch(testlabels, collect(0:9));

#functions to log information
function log_train(pred, y)
    @info "train/vals" loss=loss_fn(pred, y) acc=accuracy(pred, y)
end
function log_val()
    params_vec, _ = Flux.destructure(model)
    @info "train/weights" model=params_vec log_step_increment=0
    @info "test/vals" loss=loss_fn(model(testdata), testlabels) acc=accuracy(model(testdata), testlabels)
end

# trainloader = Flux.DataLoader((data=traindata, label=trainlabels), batchsize=100, shuffle=true, buffer=true, parallel=true) |> gpu ;
# testloader = Flux.DataLoader((data=testdata, label=testlabels), batchsize=100, shuffle=false, buffer=true) |> gpu ;

# trainloader = Flux.DataLoader((data=traindata, label=trainlabels), batchsize=100) |> gpu;
# testloader = Flux.DataLoader((data=testdata, label=testlabels), batchsize=100) |> gpu;
# simple dumb dataloader to isolate if the problem is related to the Flux.DataLoader
function batch(data, batchsize)
    return eachslice(reshape(data, size(data)[1:end-1]..., trunc(Int, size(data)[end]/batchsize), :); dims=ndims(data)+1)
end
bs100 = data -> batch(data, 100)
trainloader = zip(bs100(traindata), bs100(trainlabels)) |> gpu;
testloader = zip(bs100(testdata), bs100(testlabels)) |> gpu;

#Train
for epoch in 1:15
    println("epoch $epoch")
    for (x, y) in trainloader
        loss, grads = Flux.withgradient(model) do m
            pred = m(x)
            ChainRulesCore.ignore_derivatives() do 
                with_logger(train_logger) do 
                    log_train(pred, y)
                end
            end
            loss_fn(pred, y)
        end
        Flux.update!(opt, model, grads[1])
    end
    with_logger(val_logger) do 
        log_val()
    end
end