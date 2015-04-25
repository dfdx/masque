
using Images
using ImageView
using HDF5, JLD
using LIBSVM
using DecisionTree
using MLBase
using Boltzmann
using NaiveBayes

include("constants.jl")
include("interp.jl")
include("shape.jl")
include("data.jl")
include("analysis.jl")
include("fit.jl")


function test_raw()
    img_dat, lab_dat, nzs = read_aligned(labeled_only=true)
    
    X = img_dat
    y = lab_dat

    p, n = size(X)
    train_frac = 0.9
    k = int(floor(train_frac * n))
    idxs = randperm(n)
    train = idxs[1:k]
    test = idxs[k+1:end]
        
    m = svmtrain(y[train], X[:, train], kernel_type=LIBSVM.Linear)

    accuracy = countnz(svmpredict(m, X[:,test])[1] .== y[test]) / countnz(test)
    return accuracy
end

function test_compressed()
    img_dat, lab_dat, nzs = read_aligned(labeled_only=true)
    rbm, nzs2 = load_model()
    
    X = transform(rbm, img_dat)
    y = lab_dat

    p, n = size(X)
    train_frac = 0.9
    k = int(floor(train_frac * n))
    idxs = randperm(n)
    train = idxs[1:k]
    test = idxs[k+1:end]
        
    m = svmtrain(y[train], X[:, train], kernel_type=LIBSVM.Linear)

    accuracy = countnz(svmpredict(m, X[:,test])[1] .== y[test]) / countnz(test)
    return accuracy
end

function test_tree_raw()
    img_dat, lab_dat, nzs = read_aligned(labeled_only=true)
    rbm, nzs2 = load_model()
    
    X = img_dat
    y = lab_dat

    accuracy = nfoldCV_tree(y, X', 0.9, 10)
    return mean(accuracy)
end

function test_tree_compressed()
    img_dat, lab_dat, nzs = read_aligned(labeled_only=true)
    rbm, nzs2 = load_model()
    
    X = transform(rbm, img_dat)
    y = lab_dat

    accuracy = nfoldCV_tree(y, X', 0.9, 10)
    return mean(accuracy)
end


function dothis()
    X = rand(1000, 100) * 100
    y = rand(1:5, 100)
    for i=1:100
        println("$(i)th iteration")
        m = svmtrain(y, X)
        yhat = svmpredict(m, X)
    end
    println("Done.")
end
