#For a Colab of this example, goto https://colab.research.google.com/drive/1xfUsBn9GEqbRjBF-UX_jnGjHZNtNsMae
using TensorBoardLogger
using Flux, ChainRulesCore
using Logging
using MLDatasets
using Statistics

#create tensorboard logger
logdir = "content/log"
logger = TBLogger(logdir, tb_overwrite)

#Load data
traindata, trainlabels = FashionMNIST.traindata();
testdata, testlabels = FashionMNIST.testdata();
trainsize = size(traindata, 3);
testsize = size(testdata, 3);

#Log some images
images = TBImage(traindata[:, :, 1:10], WHN)
with_logger(logger) do #log some samples
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
    @info "train" loss=loss_fn(pred, y) acc=accuracy(pred, y)
end
function log_val()
    params_vec, _ = Flux.destructure(model)
        @info "train" model=params_vec log_step_increment=0
    @info "test" loss=loss_fn(model(testdata), testlabels) acc=accuracy(model(testdata), testlabels)
end

# trainloader = Flux.DataLoader((data=traindata, label=trainlabels), batchsize=100, shuffle=true, buffer=true, parallel=true) |> gpu ;
# testloader = Flux.DataLoader((data=testdata, label=testlabels), batchsize=100, shuffle=false, buffer=true) |> gpu ;

trainloader = Flux.DataLoader((data=traindata, label=trainlabels), batchsize=100) |> gpu ;
testloader = Flux.DataLoader((data=testdata, label=testlabels), batchsize=100) |> gpu ;

#Train
with_logger(logger) do
    for epoch in 1:15
        println("epoch $epoch")
        for (x, y) in trainloader
            loss, grads = Flux.withgradient(model) do m
                pred = m(x)
                ChainRulesCore.ignore_derivatives() do 
                    log_train(pred, y)
                end
                loss_fn(pred, y)
            end
            Flux.update!(opt, model, grads[1])
        end
        Flux.throttle(log_val(), 5)
    end
end