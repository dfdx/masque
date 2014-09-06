
using Images
using ImageView
using Milk

include("interp.jl")


function nonzero_indexes{T <: Number}(mat::Matrix{T})
    idxs = (Int, Int)[]
    for i=1:size(mat, 1)
        for j=1:size(mat, 2)
            if mat[i, j] != 0
                push!(idxs, (i, j))
            end
        end
    end
    return idxs
end


function collect_nonzeros{T <: Number}(mat::Matrix{T}, idxs::Array{(Int, Int)})
    arr = zeros(T, length(idxs))
    k = 1
    for (i, j) in idxs
        arr[k] = mat[i, j]
        k += 1
    end
    return arr
end


function map_nonzeros{T <: Number}(matsize::(Int, Int), arr::Array{T, 1},
                      idxs::Array{(Int, Int)})
    mat = zeros(T, matsize)
    k = 1
    for (i, j) in idxs
        mat[i, j] = arr[k]
        k += 1
    end
    return mat
end


IMSIZE = (96, 96)

# Create dataset consiting of nonzero pixels of face images
# Returns:
#   dat : Matrix(# of nonzeros x # of filenames) - data matrix
#   nonzero_idxs : Vector((i, j)) - nonzero pixel coordinates
function facedata(datadir="../../data/CK/faces_aligned", imsize=IMSIZE)
    filenames = readdir(datadir)
    refmat = convert(Array, imread(datadir * "/" * filenames[1]))
    refmat = imresize(refmat, imsize)
    refmat = refmat ./ 256
    nonzero_idxs = nonzero_indexes(refmat)
    dat = zeros(length(nonzero_idxs), length(filenames))
    for (i, fname) in enumerate(filenames)
        println(i, " ", fname)
        mat = convert(Array, imread(datadir * "/" * fname))
        mat = imresize(mat, imsize)
        mat = mat ./ 256
        dat[:, i] = collect_nonzeros(mat, nonzero_idxs)
    end
    return dat, nonzero_idxs
end


function wplot(W, nzs, padding=10)
    h, w = IMSIZE
    n = size(W, 1)
    rows = int(floor(sqrt(n)))
    cols = int(ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        wt = W[i, :]
        wt = reshape(wt, length(wt))
        wim = map_nonzeros(IMSIZE, wt, nzs)
        wim = wim ./ (maximum(wim) - minimum(wim))
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end
    view(dat)
    return dat
end



# REPL


using HDF5, JLD, PyPlot

flatten(a) = reshape(a, prod(size(a)))
ihist(a) = plt.hist(flatten(a), bins=100)

function run1(n_hid=512)
    dat, nzs = facedata()
    n_feat, n_samples = size(dat)
    model = GRBM(n_feat, n_hid)
    for i=1:5
        println("meta-iteration #", i)
        fit!(model, dat, n_iter=10, n_gibbs=3, lr=0.1)
        println("Sleeping for 20 seconds to cool down")
        sleep(20)
    end
    min_images = min(n_hid, 64)
    for i=1:min(3, div(n_hid, min_images))
        wplot(model.weights[(i-1)*min_images+1:i*min_images, :], nzs)
    end
    ## rest = rem(n_hid, min_images)
    ## if rest > 0
    ##     wplot(model.weights[end-rest:end, :], nzs)
    ## end
    # wplot(model.weights[div(n_hid)*min_images+1:end, :], nzs)
    return model, nzs
end

# some results
# n_hid=192, n_gibbs=3, n_meta=10 ~ 7 faces of good quality
# n_hid=192, n_gibbs=5, n_meta=10 ~ 4 faces of good quality
# n_hid=256, n_gibbs=1, n_meta=10 ~ no faces at all
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=10, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=20 ~ 12 faces of almost good quality
# n_hid=128, n_gibbs=3, n_meta=10 ~ all faces of bad quality
# n_hid=192, n_gibbs=3, n_meta=10 ~ 3 faces of a modate quality
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of moderate quality
# n_hid=320, n_gibbs=3, n_meta=10 ~ 16 faces of almost good qualiy
# n_hid=384, n_gibbs=3, n_meta=10 ~ 17 faces of almost good qualiy
# n_hid=512, n_gibbs=3, n_meta=10 ~ 22 faces of different quality
# n_hid=768, n_gibbs=3, n_meta=10 ~ 25 faces of different quality
# n_hid=1024, n_gibbs=3, n_meta=10 ~ 21 face of different quality
# n_hid=768, n_gibbs=3, n_meta=10, sigma=0.001 ~ almost all faces!
# n_hid=512, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good, but many similar
# n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good
# n_hid=128, n_gibbs=3, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# n_hid=128, n_gibbs=10, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# brbm: n_hid=768, n_gibbs=3, n_meta=10, lr=0.01 ~ 20 good
# grbm: n_hid=512, n_gibbs=3, n_meta=5, lr=0.1 ~ 


function load_model(filename="session.jld")
    d = load(filename)
    return d["m"], d["nzs"]
end



## if isinteractive()
##     im = imread("../../data/CK/faces_aligned/S005_001_00000010.png")
##     mat = convert(Array, im)
## end
